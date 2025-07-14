# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union, Optional

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class IterativeRefineDetector(SingleStage3DDetector):

    def __init__(self,
            backbone: ConfigType,
            adaptor: OptConfigType = None,
            neck: OptConfigType = None,
            bbox_head: OptConfigType = None,
            ir_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            data_preprocessor: OptConfigType = None,
            predict_with_ir: bool = False
    ) -> None:
        super(IterativeRefineDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        
        self.adaptor = MODELS.build(adaptor) if adaptor is not None else None

        self.__predict_with_ir = predict_with_ir

        if ir_head is not None:
            # update train and test cfg here for now
            ir_train_cfg = train_cfg.ir_head if train_cfg is not None else None
            ir_head.update(train_cfg=ir_train_cfg)
            ir_head.update(test_cfg=test_cfg.ir_head)

            self.ir_head = MODELS.build(ir_head)

    @property
    def with_adaptor(self) -> bool:
        return hasattr(self, 'adaptor') and self.adaptor is not None

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_ir(self) -> bool:
        return hasattr(self, 'ir_head') and self.ir_head is not None
    
    @property
    def predict_with_ir(self) -> bool:
        return self.with_ir and self.__predict_with_ir
    
    def loss(self, 
            batch_inputs_dict: dict,
            batch_data_samples: SampleList,
            **kwargs
    ) -> Union[dict, list]:
        feats_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()

        preds_dicts = self.bbox_head(feats_dict, **kwargs)
        if self.with_ir:
            soft_targets, ir_losses = self.ir_head.loss(preds_dicts, batch_data_samples, **kwargs)
            _losses = self.bbox_head.loss(preds_dicts, batch_data_samples, soft_targets=soft_targets, **kwargs)

            losses.update(_losses)
            losses.update(ir_losses)
        else:
            losses = self.bbox_head.loss(preds_dicts, batch_data_samples, **kwargs)

        return losses
    
    def predict(self, 
            batch_inputs_dict: dict,
            batch_data_samples: SampleList,
            **kwargs
    ) -> SampleList:
        if not self.predict_with_ir:
            return super(IterativeRefineDetector, self).predict(batch_inputs_dict, batch_data_samples, **kwargs)
        
        x = self.extract_feat(batch_inputs_dict)
        
        proposal = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        results_list = self.ir_head.predict(x, batch_data_samples, proposal=proposal, **kwargs)

        return self.add_pred_to_datasample(batch_data_samples, results_list)
    
    def _forward(self,
            batch_inputs_dict: dict,
            data_samples: OptSampleList = None,
            **kwargs
    ) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
        x = self.extract_feat(batch_inputs_dict)

        if self.with_ir:
            proposal = self.bbox_head(x, **kwargs)
            bbox = self.ir_head(x, proposal=proposal, **kwargs)
            outs = (bbox, proposal)
        else:
            bbox = self.bbox_head(x, **kwargs)
            outs = (bbox, None)

        return outs
    
    def extract_feat(self, 
            batch_inputs_dict: Dict[str, Tensor],
    ) -> Union[Tuple[Tensor], Dict[str, Tensor]]:
        
        if not self.with_adaptor:
            return super(IterativeRefineDetector, self).extract_feat(batch_inputs_dict)

        adapted_feat = self.adaptor(batch_inputs_dict)
        x = self.backbone(adapted_feat)
        if self.with_neck:
            x = self.neck(x)
        return x
