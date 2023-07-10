from typing import Any
from omegaconf import DictConfig

import torch

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        raise NotImplementedError
    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    @staticmethod
    def get_train_and_val_dataset(cfg: DictConfig):
        raise NotImplementedError