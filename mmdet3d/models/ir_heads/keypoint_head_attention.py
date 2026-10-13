import math
from typing import List, Dict, Optional, Callable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

extractor_registry: Dict[int, Callable] = {}

def keypoint_extractor(keypoint_num: int):
    def wrapper(func: Callable):
        extractor_registry[keypoint_num] = func
        return func
    return wrapper


@MODELS.register_module()
class KeypointHeadAttention(BaseModule):

    def __init__(self,
                 keypoint_num: int = 5,
                 out_stride: Optional[int] = None,
                 in_channels: int = 64,
                 d_model: int = 256,
                 fc_layers_share: dict = dict(
                     fc_channels=[512, 1024],
                     output_channels=1024,
                     dropout_ratio=0.5),
                 fc_layers_bbox: dict = dict(
                     fc_channels=[1024, 512],
                     output_channels=10,
                     dropout_ratio=0.5),
                 tasks: List[dict] = [dict(num_class=1)],
                 refine_tasks: Optional[List[int]] = None,
                 bbox_code_size: int = 10,
                 transformer_nhead: int = 8,
                 transformer_num_layers: int = 2,
                 transformer_dim_feedforward: int = 512,
                 transformer_dropout: float = 0.1,
                 norm_bbox: bool = True,
                 context_radius: Optional[float] = None,
                 context_tasks: Optional[List[int]] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        super(KeypointHeadAttention, self).__init__(init_cfg=init_cfg)

        assert keypoint_num in extractor_registry.keys(), f'Keypoint number {keypoint_num} is not supported.'
        self._keypoint_num = keypoint_num
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if out_stride is None:
            out_stride = (train_cfg.get('out_size_factor', 4)
                          if train_cfg is not None else 4)
        self._out_stride = out_stride
        self.tasks_cfg = tasks
        self.norm_bbox = norm_bbox
        if context_radius is not None and (
                not math.isfinite(context_radius) or context_radius <= 0):
            raise ValueError('context_radius must be finite and positive.')
        self.context_radius = context_radius
        self.context_tasks = context_tasks
        # Eight BEV directions on a circle; the radius is in meters, not
        # proportional to the proposal dimensions. Diagonals have unit norm.
        diagonal = 2 ** -0.5
        context_offsets = torch.tensor([
            [1., 0.], [diagonal, diagonal], [0., 1.],
            [-diagonal, diagonal], [-1., 0.],
            [-diagonal, -diagonal], [0., -1.],
            [diagonal, -diagonal],
        ])
        self.register_buffer('_context_offsets', context_offsets,
                             persistent=False)

        self.refine_tasks = refine_tasks
        self._bbox_code_size = bbox_code_size
        
        _d_model_in = in_channels + 2
        self.input_proj = nn.Linear(_d_model_in, d_model)

        num_tokens = keypoint_num + (8 if context_radius is not None else 0)
        # Context points use their own token IDs, distinct from box points.
        self.positional_encoding = nn.Embedding(num_tokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers)
        
        self.fc_layers_share = nn.ModuleList()
        self.fc_layers_bbox = nn.ModuleList()
        for task_id, task in enumerate(self.tasks_cfg):

            if self.refine_tasks is not None and task_id not in self.refine_tasks:
                # skip refinement for this task
                self.fc_layers_share.append(nn.Identity())
                self.fc_layers_bbox.append(nn.Identity())
                continue

            num_class = task['num_class']
            fc_share_input_channels = d_model + self._bbox_code_size + num_class
            self.fc_layers_share.append(self._build_fc_layers(
                input_channels=fc_share_input_channels, **fc_layers_share))
            # initialize the last layer to zero
            self.fc_layers_bbox.append(self._build_fc_layers(
                input_channels=fc_layers_share['output_channels'],
                output_channels=self._bbox_code_size,
                fc_channels=fc_layers_bbox['fc_channels'],
                dropout_ratio=fc_layers_bbox.get('dropout_ratio', None),
                init_last_layer_zero=True))

    def forward(self, 
                feat_maps: torch.Tensor,
                proposals: torch.Tensor,
                center_indexes: torch.Tensor,
                task_id: int = 0,
                scores: Optional[torch.Tensor] = None
                ) -> Tensor:
        if self.refine_tasks is not None and task_id not in self.refine_tasks:
            # skip refinement for this task
            return proposals

        B, N, _ = proposals.shape
        
        keypoint_feats = self._extract_feat(
            feat_maps, proposals, center_indexes, task_id=task_id)
        keypoint_feats = self.input_proj(keypoint_feats)

        _, _, K, C_feat = keypoint_feats.shape

        transformer_input = keypoint_feats.view(B * N, K, C_feat)

        pos_ids = torch.arange(K, device=transformer_input.device).unsqueeze(0)
        pos_embed = self.positional_encoding(pos_ids)
        transformer_input = transformer_input + pos_embed
        transformer_output = self.transformer_encoder(transformer_input)
        agg_feat = transformer_output.mean(dim=1)

        proposal_feats_reshaped = proposals.view(B * N, self._bbox_code_size)
        
        if scores is not None:
            scores_reshaped = scores.view(B * N, -1)
        else:
            num_class = self.tasks_cfg[task_id]['num_class']
            scores_reshaped = torch.zeros((B * N, num_class), device=agg_feat.device)
        feats_reshaped = torch.cat([agg_feat, proposal_feats_reshaped, scores_reshaped], dim=1)

        shared_feat = self.fc_layers_share[task_id](feats_reshaped)

        # add residual connection
        bbox_pred_offset = self.fc_layers_bbox[task_id](shared_feat)
        bbox_pred_offset = bbox_pred_offset.view(B, N, -1)
        refined_proposals = proposals + bbox_pred_offset

        return refined_proposals

    def _extract_feat(self, feat_maps: torch.Tensor, proposals: torch.Tensor,
                      center_indexes: torch.Tensor,
                      task_id: int = 0) -> torch.Tensor:
        box_feats = extractor_registry[self._keypoint_num](
            self, feat_maps, proposals, center_indexes)
        if self.context_radius is None or (
                self.context_tasks is not None
                and task_id not in self.context_tasks):
            return box_feats
        center = self._index_to_absl(
            center_indexes, proposals[..., :2], feat_maps.shape[-2:])
        context_points = (center.unsqueeze(2)
                          + self.context_radius * self._context_offsets.to(center))
        context_feats = self._sample_features(feat_maps, context_points)
        return torch.cat([box_feats, context_feats], dim=2)

    @keypoint_extractor(5)
    def _extract_feat_with_5_points(self, feat_maps: torch.Tensor, proposals: torch.Tensor, center_indexs: torch.Tensor) -> torch.Tensor:
        center = self._index_to_absl(
            center_indexs, proposals[..., :2], feat_maps.shape[-2:])
        # Proposals and output residuals stay in the detector's encoded space.
        # Only the sampling geometry needs physical dimensions in meters.
        dimensions = proposals[..., 3:6]
        if self.norm_bbox:
            dimensions = dimensions.exp()
        l, w = dimensions[..., 0], dimensions[..., 1]
        rotation = F.normalize(proposals[..., 6:8], dim=-1)
        sin, cos = rotation[..., 0], rotation[..., 1]
        dx, dy = l / 2, w / 2
        corners = torch.stack([torch.stack([dx, dy],-1), torch.stack([dx,-dy],-1), torch.stack([-dx,-dy],-1), torch.stack([-dx, dy],-1)], dim=2)
        rot_mat = torch.stack([torch.stack([cos,-sin],-1), torch.stack([sin, cos],-1)], dim=2)
        corners_rot = corners @ rot_mat.transpose(-1, -2)
        corners_abs = corners_rot + center.unsqueeze(2)
        mid_points = (corners_abs + torch.roll(corners_abs, shifts=-1, dims=2)) / 2
        sample_points_abs = torch.cat([center.unsqueeze(2), mid_points], dim=2)
        return self._sample_features(feat_maps, sample_points_abs)

    def _sample_features(self, feat_maps: torch.Tensor,
                         sample_points_abs: torch.Tensor) -> torch.Tensor:
        H, W = feat_maps.shape[-2:]
        xs, ys = self._absl_to_index(sample_points_abs, (H, W))
        # Pixel-center coordinates for grid_sample(align_corners=False).
        norm_xs = 2 * (xs + 0.5) / W - 1
        norm_ys = 2 * (ys + 0.5) / H - 1
        grid = torch.stack([norm_xs, norm_ys], dim=-1)
        sampled_feats = F.grid_sample(feat_maps, grid, mode='bilinear', padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
        pos_enc_x, pos_enc_y = xs / (W - 1 + 1e-6), ys / (H - 1 + 1e-6)
        pos_enc = torch.stack([pos_enc_x, pos_enc_y], dim=-1)
        final_feats = torch.cat([sampled_feats, pos_enc], dim=-1)
        return final_feats

    def _absl_to_index(
            self, absolute: torch.Tensor,
            feature_map_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = (self._feature_map_size if feature_map_size is None
                else feature_map_size)
        vs_y, vs_x = self._voxel_size[1], self._voxel_size[0]
        pcr_y, pcr_x = self._point_cloud_range[1], self._point_cloud_range[0]
        
        grid_y = (absolute[..., 1] - pcr_y) / (vs_y * self._out_stride)
        grid_x = (absolute[..., 0] - pcr_x) / (vs_x * self._out_stride)

        grid_y = torch.clamp(grid_y, 0, H - 1)
        grid_x = torch.clamp(grid_x, 0, W - 1)
        return grid_x, grid_y

    def _index_to_absl(
            self, center_indexes: torch.Tensor,
            offset: Optional[torch.Tensor] = None,
            feature_map_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        H, W = (self._feature_map_size if feature_map_size is None
                else feature_map_size)
        dtype = offset.dtype if offset is not None else torch.float32
        grid_y = (center_indexes // W).to(dtype)
        grid_x = (center_indexes % W).to(dtype)
        if offset is not None:
            grid_x += offset[..., 0]
            grid_y += offset[..., 1]
        abs_x = grid_x * self._voxel_size[0] * self._out_stride + self._point_cloud_range[0]
        abs_y = grid_y * self._voxel_size[1] * self._out_stride + self._point_cloud_range[1]
        return torch.stack([abs_x, abs_y], dim=-1)

    @property
    def _feature_map_size(self):
        cfg = self.train_cfg
        if cfg is None or 'grid_size' not in cfg or 'out_size_factor' not in cfg:
            raise ValueError('train_cfg must contain grid_size and out_size_factor.')
        return [size // cfg['out_size_factor'] for size in cfg['grid_size'][:2]]

    @property
    def _point_cloud_range(self):
        cfg = self.train_cfg
        if cfg is None or 'point_cloud_range' not in cfg:
            raise ValueError('train_cfg must contain point_cloud_range.')
        return cfg['point_cloud_range']
    
    @property
    def _voxel_size(self):
        cfg = self.train_cfg
        if cfg is None or 'voxel_size' not in cfg:
            raise ValueError('train_cfg must contain voxel_size.')
        return cfg['voxel_size']

    def _build_fc_layers(self, 
        input_channels: int, 
        fc_channels: List[int], 
        output_channels: int, 
        dropout_ratio: Optional[float] = None,
        init_last_layer_zero: bool = False
    ) -> nn.Sequential:
        fc_layers = []
        c_in = input_channels
        for k, c_out in enumerate(fc_channels):
            fc_layers.extend([
                nn.Linear(c_in, c_out, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ])
            c_in = c_out
            if k == 0 and dropout_ratio is not None and dropout_ratio > 0:
                fc_layers.append(nn.Dropout(p=dropout_ratio))
        
        final_layer = nn.Linear(c_in, output_channels, bias=True)

        if init_last_layer_zero:
            nn.init.constant_(final_layer.weight, 0)
            nn.init.constant_(final_layer.bias, 0)

        fc_layers.append(final_layer)
        
        return nn.Sequential(*fc_layers)