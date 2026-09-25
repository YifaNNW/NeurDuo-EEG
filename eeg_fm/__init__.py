"""Public API of NeurDuo-EEG: the backbone, its configuration, the pretraining wrapper, the metadata batch and the streaming runner."""

from .config import EEGFMConfig
from .model import EEGFM
from .pretrain_config import PretrainConfig
from .pretraining import EEGFMPretrainer
from .state import EEGMetadataBatch
from .streaming import StreamRunner

__all__ = [
    "EEGFM",
    "EEGFMConfig",
    "EEGFMPretrainer",
    "EEGMetadataBatch",
    "PretrainConfig",
    "StreamRunner",
]
