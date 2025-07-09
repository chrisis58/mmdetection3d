# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Optional

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class IterativeRefineDetector(SingleStage3DDetector):

    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        ir_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super(IterativeRefineDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        if ir_head is not None:
            # update train and test cfg here for now
            ir_train_cfg = train_cfg.ir_head if train_cfg is not None else None
            ir_head.update(train_cfg=ir_train_cfg)
            ir_head.update(test_cfg=test_cfg.ir_head)

            self.__predicate_with_ir = ir_head.get('predicate_with_ir', False)

            self.ir_head = MODELS.build(ir_head)

    @property
    def with_ir(self) -> bool:
        return hasattr(self, 'ir_head') and self.ir_head is not None
    
    @property
    def predicate_with_ir(self) -> bool:
        return self.__predicate_with_ir
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        feats_dict = self.extract_feat(batch_inputs_dict)

        losses = dict()

        if self.with_ir:
            bbox = self.bbox_head(feats_dict, batch_data_samples, **kwargs)
            bbox_ir, ir_losses = self.ir_head.loss(feats_dict, batch_data_samples, bbox=bbox, **kwargs)
            _losses = self.bbox_head.loss(feats_dict, batch_data_samples, bbox=bbox_ir, **kwargs)

            losses.update(_losses)
            losses.update(ir_losses)
        else:
            bbox = self.bbox_head(feats_dict, batch_data_samples, **kwargs)
            losses = self.bbox_head.loss(feats_dict, batch_data_samples, bbox=bbox, **kwargs)

        return losses
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        
        x = self.extract_feat(batch_inputs_dict)
        
        if self.predicate_with_ir:
            bbox = self.bbox_head.predict(x, batch_data_samples, **kwargs)
            results_list = self.ir_head.predict(x, batch_data_samples, bbox=bbox, **kwargs)

            predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        else:
            results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
            predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions
    
    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
        x = self.extract_feat(batch_inputs_dict)

        if self.with_ir:
            bbox = self.bbox_head(x, data_samples, **kwargs)
            bbox_ir = self.ir_head(x, data_samples, bbox=bbox, **kwargs)
            outs = (bbox_ir, bbox)
        else:
            bbox = self.bbox_head(x, data_samples, **kwargs)
            outs = (bbox, None)

        return outs
