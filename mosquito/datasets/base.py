from typing import Any, Optional
from omegaconf import DictConfig

import torch

class BaseDataset(torch.utils.data.Dataset):
    word_to_integer = {'aegypti': 0, 'albopictus': 1, 'anopheles': 2, 'culex': 3, 'culiseta': 4, 'japonicus/koreicus': 5}
    integer_to_word = {v: k for k, v in word_to_integer.items()}
    
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