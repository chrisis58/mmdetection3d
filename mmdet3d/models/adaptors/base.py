from abc import ABCMeta, abstractmethod

from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from mmdet3d.registry import MODELS

class BaseAdaptor(BaseModule, metaclass=ABCMeta):

    def __init__(self,
            encoder: ConfigType,
            middle_encoder: OptConfigType = None,
            *args, **kwargs
    ) -> None:
        super(BaseAdaptor, self).__init__(*args, **kwargs)

        self.encoder = MODELS.build(encoder)
        if middle_encoder is not None:
            self.middle_encoder = MODELS.build(middle_encoder)

    @property
    def with_middle_encoder(self) -> bool:
        return hasattr(self, 'middle_encoder') and self.middle_encoder is not None

    @abstractmethod
    def forward(self, *args, **kwargs): ...
