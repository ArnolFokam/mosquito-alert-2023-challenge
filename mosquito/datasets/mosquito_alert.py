from functools import lru_cache
import logging
import os
import PIL.Image
import numpy as np
import pandas as pd
from typing import Any, Optional
from mosquito.datasets.base import BaseDataset


class MosquitoAlertv0(BaseDataset):
    images_folder: str = "train_images/"
    labels_csv_file: str = "train.csv"
    
    def __init__(self, 
                 cfg,
                 transform: Optional[callable] = None) -> None:
        self.cfg = cfg
        self.transform = transform
        
        # get the csv annotations file
        self.annotations_df = pd.read_csv(os.path.join(cfg.data_dir, self.labels_csv_file))
        
        # extract the annotations
        filenames = self.annotations_df["img_fName"].values
        bboxes = self.annotations_df[["bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr"]].values
        labels = self.annotations_df["class_label"].values
        
        # convert labels to integers
        word_to_integer = {}

        # Assign unique integer values to each word
        for i, word in enumerate(np.unique(labels)):
            word_to_integer[word] = i
            
        # Convert all words to integers
        labels = list(map(lambda x: word_to_integer[x], labels))
        
        # convert all images to absolute paths
        filenames = list(map(lambda x: os.path.join(cfg.data_dir, self.images_folder, x), filenames))
        
        # zip all the data together
        self.data = list(zip(filenames, bboxes, labels))
        
    @lru_cache(maxsize=1000000)
    def load_image(self, filename):
        return PIL.Image.open(filename).convert("RGB")
        
    def __getitem__(self, index) -> Any:
        filename, bbox, label = self.data[index]
        image = self.load_image(filename)
        
        if self.transform is not None:
            image, bbox = self.transform(image, bbox)
            
        return image, bbox, label
    
    def __len__(self) -> int:
        return len(self.data)
    
    @staticmethod
    def get_train_and_val_dataset(cfg):
        dataset = MosquitoAlertv0(cfg)
        return dataset, None
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
        
        