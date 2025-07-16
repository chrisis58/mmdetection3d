from typing import List, Dict, Optional, Callable, Tuple

import torch
from torch import nn

from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

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
            pc_start: List[float] = [-51.2, -51.2],
            voxel_size: List[float] = [0.2, 0.2],
            out_stride: int = 4,
            in_channels: int = 128,
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
                output_channels=7,
                dropout_ratio=0.5),
            loss_bbox: dict = dict(
                type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
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
        self.fc_layers_share = self._build_fc_layers(
            input_channels=in_channels * keypoint_num,
            fc_channels=fc_layers_share['fc_channels'],
            output_channels=__in_channels,
            dropout_ratio=fc_layers_share.get('dropout_ratio', None))
        
        self.fc_layers_cls = self._build_fc_layers(
            input_channels=__in_channels,
            fc_channels=fc_layers_cls['fc_channels'],
            output_channels=fc_layers_cls['output_channels'],
            dropout_ratio=fc_layers_cls.get('dropout_ratio', None))
        
        self.loss_cls = MODELS.build(loss_cls)

        self.fc_layers_bbox = self._build_fc_layers(
            input_channels=__in_channels,
            fc_channels=fc_layers_bbox['fc_channels'],
            output_channels=fc_layers_bbox['output_channels'],
            dropout_ratio=fc_layers_bbox.get('dropout_ratio', None))
        
        self.loss_bbox = MODELS.build(loss_bbox)

    def forward(self, 
            x: List[torch.Tensor],
            proposals: List[InstanceData],
    ) -> Dict:
        ir_cls_score, ir_bbox_pred = multi_apply(self._forward_single, x[0], proposals)

        return dict(
            ir_cls_scores=ir_cls_score,
            ir_bbox_preds=ir_bbox_pred)
    
    def _forward_single(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._extract_feat(feat_map, proposal)

        shared_feat = self.fc_layers_share(feat)

        cls_score = self.fc_layers_cls(shared_feat)
        bbox_pred = self.fc_layers_bbox(shared_feat)

        return cls_score, bbox_pred
    
    def loss(self,
            preds_dict: Dict[str, torch.Tensor],
            batch_data_samples: List[dict]
    ) -> Dict[str, torch.Tensor]:
        cls_scores = preds_dict['ir_cls_scores']
        bbox_preds = preds_dict['ir_bbox_preds']

        gt_labels = [data_sample['gt_labels'] for data_sample in batch_data_samples]
        gt_bboxes = [data_sample['gt_bboxes'] for data_sample in batch_data_samples]

        gt_labels = torch.stack(gt_labels, dim=0)
        gt_bboxes = torch.stack(gt_bboxes, dim=0)

        return self._loss(cls_scores, bbox_preds, gt_labels, gt_bboxes)
    
    def _loss(self, 
            cls_scores: torch.Tensor,
            bbox_preds: torch.Tensor,
            gt_labels: torch.Tensor,
            gt_bboxes: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        losses = dict()
        losses['loss_cls'] = self.loss_cls(cls_scores, gt_labels)
        losses['loss_bbox'] = self.loss_bbox(bbox_preds, gt_bboxes)
        return losses
    
    def predict(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> List[InstanceData]:
        ...
    
    def _extract_feat(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> List[torch.Tensor]:
        return extractor_registry[self._keypoint_num](self, feat_map, proposal)

    @keypoint_extractor(5)
    def _extract_feat_with_5_points(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> List[torch.Tensor]:
        raise NotImplementedError()
        
    
    @keypoint_extractor(1)
    def _extract_feat_with_1_points(self,
            feat_map: torch.Tensor,
            proposal: InstanceData
    ) -> torch.Tensor:
        bboxes = proposal.bboxes_3d.tensor

        points = bboxes[:, :3]
        points = points.view(points.size(0), -1)
        
        xs, ys = self._absl_to_relative(points)

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
