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
                 out_stride: int = 4,
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
                 tasks: int = 6,
                 transformer_nhead: int = 8,
                 transformer_num_layers: int = 2,
                 transformer_dim_feedforward: int = 512,
                 transformer_dropout: float = 0.1,
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
        self._out_stride = out_stride
        
        
        _d_model_in = in_channels + 2
        self.input_proj = nn.Linear(_d_model_in, d_model)

        self.positional_encoding = nn.Embedding(keypoint_num, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers
        )
        
        fc_share_input_channels = d_model + 10
        bbox_input_channels = fc_layers_share['output_channels']

        self.fc_layers_share = nn.ModuleList()
        self.fc_layers_bbox = nn.ModuleList()
        for i in range(tasks):
            self.fc_layers_share.append(self._build_fc_layers(
                input_channels=fc_share_input_channels, **fc_layers_share))
            self.fc_layers_bbox.append(self._build_fc_layers(
                input_channels=bbox_input_channels, **fc_layers_bbox))

    def forward(self, 
                feat_maps: torch.Tensor,
                proposals: torch.Tensor,
                center_indexes: torch.Tensor,
                task_id: int = 0,
                ) -> Tensor:
        B, N, _ = proposals.shape
        
        keypoint_feats = self._extract_feat(feat_maps, proposals, center_indexes)
        keypoint_feats = self.input_proj(keypoint_feats)

        _, _, K, C_feat = keypoint_feats.shape

        transformer_input = keypoint_feats.view(B * N, K, C_feat)

        pos_ids = torch.arange(K, device=transformer_input.device).unsqueeze(0)
        pos_embed = self.positional_encoding(pos_ids)
        transformer_input = transformer_input + pos_embed

        transformer_output = self.transformer_encoder(transformer_input)

        agg_feat = transformer_output.mean(dim=1)

        proposal_feats_reshaped = proposals.view(B * N, 10)
        feats_reshaped = torch.cat([agg_feat, proposal_feats_reshaped], dim=1)

        shared_feat = self.fc_layers_share[task_id](feats_reshaped)
        bbox_pred_reshaped = self.fc_layers_bbox[task_id](shared_feat)

        bbox_pred = bbox_pred_reshaped.view(B, N, -1)

        return bbox_pred

    def _extract_feat(self, feat_maps: torch.Tensor, proposals: torch.Tensor, center_indexes: torch.Tensor) -> torch.Tensor:
        return extractor_registry[self._keypoint_num](self, feat_maps, proposals, center_indexes)

    @keypoint_extractor(5)
    def _extract_feat_with_5_points(self, feat_maps: torch.Tensor, proposals: torch.Tensor, center_indexs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_maps.shape
        N = proposals.shape[1]
        center = self._index_to_absl(center_indexs, proposals[..., :2])
        l, w, sin, cos = proposals[..., 3], proposals[..., 4], proposals[..., 6], proposals[..., 7]
        dx, dy = l / 2, w / 2
        corners = torch.stack([torch.stack([dx, dy],-1), torch.stack([dx,-dy],-1), torch.stack([-dx,-dy],-1), torch.stack([-dx, dy],-1)], dim=2)
        rot_mat = torch.stack([torch.stack([cos,-sin],-1), torch.stack([sin, cos],-1)], dim=2)
        corners_rot = corners @ rot_mat
        corners_abs = corners_rot + center.unsqueeze(2)
        mid_points = (corners_abs + torch.roll(corners_abs, shifts=-1, dims=2)) / 2
        sample_points_abs = torch.cat([center.unsqueeze(2), mid_points], dim=2)
        xs, ys = self._absl_to_index(sample_points_abs)
        norm_xs, norm_ys = (xs / (W - 1 + 1e-6)) * 2 - 1, (ys / (H - 1 + 1e-6)) * 2 - 1
        grid = torch.stack([norm_xs, norm_ys], dim=-1)
        sampled_feats = F.grid_sample(feat_maps, grid, mode='bilinear', padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
        pos_enc_x, pos_enc_y = xs / (W - 1 + 1e-6), ys / (H - 1 + 1e-6)
        pos_enc = torch.stack([pos_enc_x, pos_enc_y], dim=-1)
        final_feats = torch.cat([sampled_feats, pos_enc], dim=-1)
        return final_feats

    def _absl_to_index(self, absolute: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = self._feature_map_size
        vs_y, vs_x = self._voxel_size[1], self._voxel_size[0]
        pcr_y, pcr_x = self._point_cloud_range[1], self._point_cloud_range[0]
        
        grid_y = (absolute[..., 1] - pcr_y) / (vs_y * self._out_stride)
        grid_x = (absolute[..., 0] - pcr_x) / (vs_x * self._out_stride)

        grid_y = torch.clamp(grid_y, 0, H - 1)
        grid_x = torch.clamp(grid_x, 0, W - 1)
        return grid_x, grid_y

    def _index_to_absl(self, center_indexes: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = self._feature_map_size
        grid_y = center_indexes // W + 0.5
        grid_x = center_indexes % W + 0.5
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

    def _build_fc_layers(self, input_channels: int, fc_channels: List[int], output_channels: int, dropout_ratio: Optional[float] = None) -> nn.Sequential:
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
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)