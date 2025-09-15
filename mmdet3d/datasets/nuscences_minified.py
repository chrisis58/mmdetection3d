from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS

@DATASETS.register_module()
class MinifiedNuScenesDataset(NuScenesDataset):
    """
    Dataset wrapper for overfitting test on a small subset of NuScenes dataset.
    It truncates the dataset to the first `num_samples` samples if specified.
    """
    def __init__(self, *args, **kwargs):
        num_samples = kwargs.pop('num_samples', None)
        
        super().__init__(*args, **kwargs)
        
        if num_samples is not None:
            if hasattr(self, 'data_list'):
                self.data_list = self.data_list[:num_samples]
                print(f'\nDataset truncated to the first {num_samples} samples for overfitting test.\n')
            else:
                self.data_infos = self.data_infos[:num_samples]
                print(f'\nDataset truncated to the first {num_samples} samples for overfitting test.\n')