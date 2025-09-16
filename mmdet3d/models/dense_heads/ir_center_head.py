from typing import List, Optional, Union, Dict, Tuple

from torch import Tensor
import torch

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from mmdet3d.models.utils import (clip_sigmoid)

from .centerpoint_head import CenterHead

@MODELS.register_module()
class IRCenterHead(CenterHead):

    def __init__(self,
            in_channels: Union[List[int], int] = [128],
            warmup_epochs: int = 3,
            tasks: Optional[List[dict]] = None,
            bbox_coder: Optional[dict] = None,
            common_heads: dict = dict(),
            loss_cls: dict = dict(
                type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox: dict = dict(
                type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
            separate_head: dict = dict(
                type='mmdet.SeparateHead',
                init_bias=-2.19,
                final_kernel=3),
            ir_head: Optional[dict] = dict(
                type='KeypointHead',
            ),
            share_conv_channel: int = 64,
            num_heatmap_convs: int = 2,
            conv_cfg: dict = dict(type='Conv2d'),
            norm_cfg: dict = dict(type='BN2d'),
            bias: str = 'auto',
            norm_bbox: bool = True,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
            **kwargs
    ) -> None:
        super(IRCenterHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            num_heatmap_convs=num_heatmap_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            norm_bbox=norm_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
        if ir_head is not None:
            ir_head['train_cfg'] = train_cfg
            self.ir_head = MODELS.build(ir_head)
        
        self._epoch = 0

        assert warmup_epochs >= 0, 'warmup_epochs must be non-negative.'
        self._warmup_epochs = warmup_epochs
    
    def with_ir(self) -> bool:
        return hasattr(self, 'ir_head') and self.ir_head is not None
    
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def loss(self, 
            pts_feats: List[Tensor],
            batch_data_samples: List[Det3DDataSample],
            *args,
            **kwargs
    ) -> Dict[str, Tensor]:
        
        if not self.with_ir:
            # 如果没有IR head，直接调用父类的loss方法
            return super().loss(pts_feats, batch_data_samples, *args, **kwargs)
        
        preds_dicts = self(pts_feats)

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)

        losses = self._loss_by_soft_targets(pts_feats, preds_dicts, batch_gt_instance_3d)

        return losses

    def _loss_by_soft_targets(
            self,
            pts_feats: List[Tensor],
            preds_dicts: Tuple[List[dict]],
            batch_gt_instance_3d: List[InstanceData],
            *args,
            **kwargs
    ) -> Dict[str, Tensor]:
        """Calculate loss using soft targets."""
        heatmaps, anno_boxes, inds, masks = self.get_targets(batch_gt_instance_3d)

        # TODO: find a better way to get the shared features
        shared_feat = self.shared_conv(pts_feats[0])

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            
            # soft target <-> gt
            soft_target = self.ir_head(
                shared_feat,
                pred,
                ind,
                task_id=task_id)
            loss_soft_gt = self.loss_bbox(
                soft_target, target_box, bbox_weights, avg_factor=(num + 1e-4))

            # conditional soft target loss: pred <-> soft_target
            loss_pred_soft = torch.tensor(0.0, device=pred.device)
            with torch.no_grad():
                should_train_soft = (self._epoch >= self._warmup_epochs and loss_soft_gt < loss_bbox * 1.1)
            if should_train_soft:
                loss_pred_soft = self.loss_bbox(
                    pred, soft_target.detach(), bbox_weights, avg_factor=(num + 1e-4))

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'task{task_id}.loss_pred_soft'] = loss_pred_soft
            loss_dict[f'task{task_id}.loss_soft_gt'] = loss_soft_gt
        return loss_dict
