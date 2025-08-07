from typing import List, Dict, Optional, Callable, Tuple

import torch
from torch import Tensor, nn

from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from ..middle_encoders.voxel_set_abstraction import bilinear_interpolate_torch

extractor_registry: Dict[int, Callable] = {}

def keypoint_extractor(keypoint_num: int):
    def wrapper(func: Callable):
        extractor_registry[keypoint_num] = func
        return func
    return wrapper

@MODELS.register_module()
class KeypointHead(BaseModule):

    def __init__(self,
            keypoint_num: int = 5,
            out_stride: int = 4,
            in_channels: int = 10,
            fc_layers_share: dict = dict(
                fc_channels=[256, 256],
                output_channels=256,
                dropout_ratio=0.5),
            fc_layers_bbox: dict = dict(
                fc_channels=[256, 256],
                output_channels=10,
                dropout_ratio=0.5),
            tasks: int = 6,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
            **kwargs
    ) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(KeypointHead, self).__init__(init_cfg=init_cfg, **kwargs)

        assert keypoint_num in extractor_registry.keys(), f'Keypoint number {keypoint_num} is not supported. ' \
            f'Supported keypoint numbers are: {extractor_registry.keys()}.'
        self._keypoint_num = keypoint_num

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._out_stride = out_stride

        __in_channels = fc_layers_share['output_channels']

        self.fc_layers_share = []
        self.fc_layers_bbox = []
        for i in range(tasks):
            self.fc_layers_share.append(self._build_fc_layers(
                input_channels=(in_channels + 2) * keypoint_num + 10,  # 10 is for the proposal features, 2 for position encoding
                fc_channels=fc_layers_share['fc_channels'],
                output_channels=__in_channels,
                dropout_ratio=fc_layers_share.get('dropout_ratio', None)))
            
            self.fc_layers_bbox.append(self._build_fc_layers(
                input_channels=__in_channels,
                fc_channels=fc_layers_bbox['fc_channels'],
                output_channels=fc_layers_bbox['output_channels'],
                dropout_ratio=fc_layers_bbox.get('dropout_ratio', None)))
        
        self.fc_layers_share = nn.ModuleList(self.fc_layers_share)
        self.fc_layers_bbox = nn.ModuleList(self.fc_layers_bbox)

    def forward(self, 
            feat_maps: torch.Tensor,
            proposals: torch.Tensor,
            center_indexes: torch.Tensor,
            task_id: int = 0,
    ) -> Tensor:
        """
        
        :feat_map: Tensor
            Feature map with shape [batch, channel, height, width].
            e.g. [4, 384, 128, 128].
        :proposals: Tensor
            Proposals with shape [batch, num_proposals, 10].
            e.g. [4, 500, 10].
        :center_indexes: Tensor
            Center indexes with shape [batch, num_proposals].
            e.g. [4, 500].
        :masks: Tensor
            Masks with shape [batch, num_proposals].
            e.g. [4, 500].
        :return: Tensor
            Target predictions with shape [batch, num_proposals, 10].
            e.g. [4, 500, 10].
        """
        batch = feat_maps.size(0)
        assert proposals.size(0) == batch, \
            f'Batch size of feature map {feat_maps.size(0)} ' \
            f'and proposals {proposals.size(0)} should be the same.'
        
        preds = []

        for i in range(batch):
            feat_map = feat_maps[i]  # [channel, height, width]
            proposal = proposals[i]  # [num_proposals, 10]
            center_index = center_indexes[i]  # [num_proposals]

            feat_map = feat_map.to(proposal.device)

            feat = self._extract_feat(feat_map, proposal, center_index)
            feat = torch.cat([feat, proposal], dim=1)

            shared_feat = self.fc_layers_share[task_id](feat)
            bbox_pred = self.fc_layers_bbox[task_id](shared_feat)

            preds.append(bbox_pred)

        return torch.stack(preds)

    def predict(self,
            feat_map: torch.Tensor,
            proposal: torch.Tensor
    ) -> List[InstanceData]:
        ...
    
    def _extract_feat(self,
            feat_map: torch.Tensor,
            proposal: torch.Tensor,
            center_index: torch.Tensor
    ) -> torch.Tensor:
        return extractor_registry[self._keypoint_num](self, feat_map, proposal, center_index)

    @keypoint_extractor(5)
    def _extract_feat_with_5_points(self,
            feat_map: torch.Tensor,
            proposal: torch.Tensor,
            center_index: torch.Tensor
    ) -> torch.Tensor:
        feat_map = feat_map.permute(1, 2, 0)  # [height, width, channel]

        center = self._index_to_absl(center_index, proposal[:, :2])

        l = proposal[:, 3]  # length
        w = proposal[:, 4]  # width

        sin = proposal[:, 6]
        cos = proposal[:, 7]

        dx = l / 2
        dy = w / 2
        corners = torch.stack([
            torch.stack([ dx,  dy], dim=1),
            torch.stack([ dx, -dy], dim=1),
            torch.stack([-dx, -dy], dim=1),
            torch.stack([-dx,  dy], dim=1)
        ], dim=1)  # [num_proposals, 4, 2]

        rot_mat = torch.stack([
            torch.stack([ cos, -sin], dim=1),
            torch.stack([ sin,  cos], dim=1)
        ], dim=2)  # [num_proposals, 2, 2]

        corners_rot = torch.einsum('bni,bij->bnj', corners, rot_mat)  # [num_proposals, 4, 2]
        corners_abs = corners_rot + center.unsqueeze(1)  # [num_proposals, 4, 2]

        mid_points = (corners_abs + torch.roll(corners_abs, shifts=-1, dims=1)) / 2  # [num_proposals, 4, 2]

        sample_points = torch.cat([center.unsqueeze(1), mid_points], dim=1)  # [num_proposals, 5, 2]
        xs, ys = self._absl_to_index(sample_points.reshape(-1, 2))
        xs = xs.view(sample_points.shape[0], sample_points.shape[1])
        ys = ys.view(sample_points.shape[0], sample_points.shape[1])

        H, W = self._feature_map_size
        norm_x = xs.float() / (W - 1)
        norm_y = ys.float() / (H - 1)
        pos_enc = torch.cat([norm_x.unsqueeze(-1), norm_y.unsqueeze(-1)], dim=-1)
        pos_enc = pos_enc.view(sample_points.shape[0], -1) # [num_proposals, 10]

        feats = []
        for i in range(5):
            feat_i = bilinear_interpolate_torch(feat_map, xs[:, i], ys[:, i])  # [num_proposals, in_channels]
            # feat_i = self._gather_feat(feat_map, xs[:, i], ys[:, i])  # [num_proposals, in_channels]

            feat_i = torch.cat([feat_i, pos_enc[:, 2 * i: 2 * i + 2]], dim=1)  # [num_proposals, in_channels + 2]

            feats.append(feat_i)  # [num_proposals, in_channels + 2]
        feats = torch.cat(feats, dim=1)  # [num_proposals, (in_channels + 2) * 5]

        return feats
    
    @keypoint_extractor(1)
    def _extract_feat_with_1_points(self,
            feat_map: torch.Tensor,
            proposal: torch.Tensor,
            center_index: torch.Tensor
    ) -> torch.Tensor:
        feat_map = feat_map.permute(1, 2, 0)  # [height, width, channel]

        center = self._index_to_absl(center_index, proposal[:, :2])
        xs, ys = self._absl_to_index(center)

        H, W = self._feature_map_size
        norm_x = xs.float() / (W - 1)
        norm_y = ys.float() / (H - 1)
        pos_enc = torch.cat([norm_x.unsqueeze(-1), norm_y.unsqueeze(-1)], dim=-1)
        pos_enc = pos_enc.view(center.shape[0], -1)

        feat = bilinear_interpolate_torch(feat_map, xs, ys)
        # feat = self._gather_feat(feat_map, xs, ys)  # [num_proposals, in_channels]
        feat = torch.cat([feat, pos_enc], dim=1)  # [num_proposals, in_channels + 2]

        return feat
    
    def _gather_feat(self, feat: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        H, W, C = feat.shape

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        return feat[y, x]  # [num_proposals, channel]
    
    def _absl_to_index(self, absolute: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = self._feature_map_size
        grid_y = (absolute[..., 1] - self._point_cloud_range[1]) / (self._voxel_size[1] * self._out_stride)
        grid_x = (absolute[..., 0] - self._point_cloud_range[0]) / (self._voxel_size[0] * self._out_stride)

        grid_y = torch.clamp(grid_y, 0, H - 1).to(torch.int32)
        grid_x = torch.clamp(grid_x, 0, W - 1).to(torch.int32)

        return grid_x, grid_y

    def _index_to_absl(self, center_indexes: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = self._feature_map_size
        grid_y = center_indexes // W + 0.5
        grid_x = center_indexes % W + 0.5

        if offset is not None:
            assert offset.shape[-1] == 2, 'Offset must have shape [N, 2].'
            grid_x = grid_x + offset[..., 0]
            grid_y = grid_y + offset[..., 1]

        abs_x = grid_x * self._voxel_size[0] * self._out_stride + self._point_cloud_range[0]
        abs_y = grid_y * self._voxel_size[1] * self._out_stride + self._point_cloud_range[1]

        return torch.stack([abs_x, abs_y], dim=-1)
    
    @property
    def _feature_map_size(self):
        if self.train_cfg is None or 'grid_size' not in self.train_cfg or 'out_size_factor' not in self.train_cfg:
            raise ValueError('train_cfg must contain grid_size and out_size_factor.')
        return [size // self.train_cfg['out_size_factor'] for size in self.train_cfg['grid_size'][:2]]

    @property
    def _point_cloud_range(self):
        if self.train_cfg is None or 'point_cloud_range' not in self.train_cfg:
            raise ValueError('train_cfg must contain point_cloud_range.')
        return self.train_cfg['point_cloud_range']
    
    @property
    def _voxel_size(self):
        if self.train_cfg is None or 'voxel_size' not in self.train_cfg:
            raise ValueError('train_cfg must contain voxel_size.')
        return self.train_cfg['voxel_size']

    def _build_fc_layers(self, 
            input_channels: int,
            fc_channels: List[int],
            output_channels: int,
            dropout_ratio: Optional[float] = None
    ) -> nn.Sequential:

        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_channels.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_channels[k], bias=False),
                nn.BatchNorm1d(fc_channels[k]),
                nn.ReLU(),
            ])
            c_in = fc_channels[k]
            if k == 0 and dropout_ratio is not None and dropout_ratio >= 0:
                fc_layers.append(nn.Dropout(p=dropout_ratio))
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)
