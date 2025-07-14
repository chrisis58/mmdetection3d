from typing import List, Optional, Union, Dict, Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from .centerpoint_head import CenterHead

@MODELS.register_module()
class IRCenterHead(CenterHead):

    def __init__(self,
            in_channels: Union[List[int], int] = [128],
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
        
    def loss(self, 
            preds_dicts: Tuple[List[dict]],
            batch_data_samples: List[Det3DDataSample],
            soft_targets: Optional[List[Tensor]] = None,
            *args,
            **kwargs
    ) -> Dict[str, Tensor]:

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
        losses = self.loss_by_feat(preds_dicts, batch_gt_instance_3d)

        if soft_targets is not None:
            soft_targets_losses = self._loss_by_soft_targets(preds_dicts, soft_targets)
            losses.update(soft_targets_losses)

        return losses

    def _loss_by_soft_targets(
            self,
            preds_dicts: Tuple[List[dict]],
            soft_targets: List[Tensor],
            *args,
            **kwargs
    ) -> Dict[str, Tensor]:
        """Calculate loss using soft targets."""
        return dict()
            
            
