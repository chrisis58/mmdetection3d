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
            keypoint_num: int = 1,
            pc_start: List[float] = [-51.2, -51.2],
            voxel_size: List[float] = [0.2, 0.2],
            out_stride: int = 4,
            in_channels: int = 384,
            fc_layers_share: dict = dict(
                fc_channels=[256, 256],
                output_channels=256,
                dropout_ratio=0.5),
            fc_layers_cls: dict = dict(
                fc_channels=[256, 256],
                output_channels=10,
                dropout_ratio=0.5),
            loss_cls: dict = dict(
                type='mmdet.CrossEntropyLoss', reduction='mean'),
            fc_layers_bbox: dict = dict(
                fc_channels=[256, 256],
                output_channels=10,
                dropout_ratio=0.5),
            loss_bbox: dict = dict(
                type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
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

        self._pc_start = pc_start
        self._voxel_size = voxel_size
        self._out_stride = out_stride

        __in_channels = fc_layers_share['output_channels']

        self.fc_layers_share = []
        self.fc_layers_bbox = []
        for i in range(tasks):
            self.fc_layers_share.append(self._build_fc_layers(
                input_channels=in_channels * keypoint_num + 10,  # 10 is for the proposal features
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
            task_id: int = 0,
    ) -> Tensor:
        """
        
        :feat_map: Tensor
            Feature map with shape [batch, channel, height, width].
            e.g. [4, 384, 128, 128].
        :proposals: Tensor
            Proposals with shape [batch, num_proposals, 10].
            e.g. [4, 500, 10].
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

            feat_map = feat_map.to(proposal.device)

            feat = self._extract_feat(feat_map, proposal)
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
            proposal: torch.Tensor
    ) -> torch.Tensor:
        return extractor_registry[self._keypoint_num](self, feat_map, proposal)

    @keypoint_extractor(5)
    def _extract_feat_with_5_points(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> torch.Tensor:
        raise NotImplementedError()
        
    
    @keypoint_extractor(1)
    def _extract_feat_with_1_points(self,
            feat_map: torch.Tensor,
            proposal: torch.Tensor
    ) -> torch.Tensor:
        center = proposal[:, :2]  # [batch, 2] 取前2个维度作为中心点
        feat_map = feat_map.permute(1, 2, 0)  # [height, width, channel]
        
        xs, ys = self._absl_to_relative(center)

        return bilinear_interpolate_torch(feat_map, xs, ys)
    

    def _absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self._pc_start[0]) / self._voxel_size[0] / self._out_stride 
        a2 = (absolute[..., 1] - self._pc_start[1]) / self._voxel_size[1] / self._out_stride 

        return a1, a2

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
