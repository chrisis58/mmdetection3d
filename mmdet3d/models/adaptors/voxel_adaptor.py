from typing import Dict

from torch import Tensor

from mmdet3d.utils import ConfigType, OptConfigType
from mmdet3d.registry import MODELS

from .base import BaseAdaptor

@MODELS.register_module()
class VoxelAdaptor(BaseAdaptor):

    def __init__(self,
            encoder: ConfigType,
            middle_encoder: OptConfigType = None,
            *args, **kwargs
    ):
        super(VoxelAdaptor, self).__init__(encoder, middle_encoder, *args, **kwargs)
        
    def forward(self, 
            batch_inputs_dict: Dict[str, Tensor],
            *args, **kwargs
    ) -> Tensor:

        voxel_dict = batch_inputs_dict.get('voxels', None)

        voxel_feat = self.encoder(voxel_dict['voxels'],
                voxel_dict['num_points'],
                voxel_dict['coors'])
        
        if not self.with_middle_encoder:
            return voxel_feat

        batch_size = voxel_dict['coors'][-1, 0] + 1
        voxel_feat = self.middle_encoder(voxel_feat, voxel_dict['coors'], batch_size)
        return voxel_feat
