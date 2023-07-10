from typing import Any
from mosquito.datasets.base import BaseDataset


class MosquitoAlertv0(BaseDataset):
    images_folder: str = "train_images"
    labels_csv_file: str = "train.csv"
    
    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def __getitem__(self, index) -> Any:
        pass
    
    def __len__(self) -> int:
        pass
    
    @staticmethod
    def get_train_and_val_dataset(cfg):
        dataset = MosquitoAlertv0(cfg)
        return dataset, None
        
        