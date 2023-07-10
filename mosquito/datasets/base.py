from typing import Any, Optional
from omegaconf import DictConfig

import torch

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    @property
    def num_classes(self) -> int:
        raise NotImplementedError
    
    @staticmethod
    def get_train_and_val_dataset(cfg: DictConfig, tranform: Optional[callable] = None):
        raise NotImplementedError
    
    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError