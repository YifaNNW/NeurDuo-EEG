"""Downstream fine-tuning: task settings, data loading, models, heads and the training loop."""

from .dataset import LoadDataset
from .dataset_specs import DATASET_SPECS, get_spec
from .net_eegfm import Model, load_model_config
from .net_seq import SeqModel, SingleEpochModel
from .trainer import Trainer

__all__ = [
    "DATASET_SPECS",
    "LoadDataset",
    "Model",
    "SeqModel",
    "SingleEpochModel",
    "Trainer",
    "get_spec",
    "load_model_config",
]
