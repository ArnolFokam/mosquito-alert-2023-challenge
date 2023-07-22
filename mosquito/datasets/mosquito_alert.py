import os
import PIL.Image
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Any, Optional
from mosquito.datasets.base import BaseDataset

# because some images are too large
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def resize_image_and_bbox(image_path, bbox, max_size):
    # Open the image using Pillow
    image = PIL.Image.open(image_path).convert("RGB")

    # Get the original width and height
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / float(original_height)

    # Calculate new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_width = int(max_size * aspect_ratio)
        new_height = max_size

    # Resize the image using the calculated dimensions
    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

    # Calculate scale factors for x and y coordinates
    x_scale = new_width / original_width
    y_scale = new_height / original_height

    # Update the bounding box coordinates
    new_bbox = [
        int(bbox[0] * x_scale),
        int(bbox[1] * y_scale),
        int(bbox[2] * x_scale),
        int(bbox[3] * y_scale)
    ]

    return resized_image, tuple(new_bbox)


class MosquitoAlertDatasetv0(BaseDataset):
    images_folder: str = "train_images/"
    labels_csv_file: str = "train.csv"
    
    def __init__(self, 
                 cfg,
                 transform: Optional[callable] = None) -> None:
        super().__init__(cfg)
        
        self.transform = transform
        
        # get the csv annotations file
        self.annotations_df = pd.read_csv(os.path.join(cfg.data_dir, self.labels_csv_file))
        
        # extract the annotations
        filenames = self.annotations_df["img_fName"].values
        bboxes = self.annotations_df[["bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr"]].values
        image_ws = self.annotations_df["img_w"].values
        image_hs = self.annotations_df["img_h"].values
        labels = self.annotations_df["class_label"].values
        
        # convert labels to integers
        self.word_to_integer = {}

        # Assign unique integer values to each word
        for i, word in enumerate(np.unique(labels)):
            self.word_to_integer[word] = i
            
        # Convert all words to integers
        labels = list(map(lambda x: self.word_to_integer[x], labels))
        
        # convert all images to absolute paths
        filenames = list(map(lambda x: os.path.join(cfg.data_dir, self.images_folder, x), filenames))
        
        # resize bounding boxes within the image dimensions
        bboxes = list(map(lambda i: 
            tuple([
                max(0, bboxes[i][0]), 
                max(0, bboxes[i][1]), 
                min(image_ws[i], bboxes[i][2]),
                min(image_hs[i], bboxes[i][3])
            ]), 
        range(len(bboxes))))
        
        # zip all the data together
        self.data = list(zip(filenames, bboxes, labels))
        
    @lru_cache(maxsize=100000)
    def load_image_and_update_bbox(self, filename, bbox):
        return resize_image_and_bbox(filename, bbox, self.cfg.load_input_size)
        
    @property
    def num_classes(self) -> int:
        return len(self.word_to_integer)
        
    def __getitem__(self, index) -> Any:
        filename, bbox, label = self.data[index]
        
        assert isinstance(bbox, tuple)
        image, bbox = self.load_image_and_update_bbox(filename, bbox)
        image = np.array(image)
        
        target = {}
        target["boxes"] = [bbox]
        target["labels"] = [label]
        target['image_id'] = filename
        
        if self.transform is not None:
            image, target = self.transform(image, target)
            
        return image, target
    
    def __len__(self) -> int:
        # return len(self.data)
        return 100
    
    @staticmethod
    def get_train_and_val_dataset(cfg, transform=None):
        dataset = MosquitoAlertDatasetv0(cfg, transform)
        return dataset, None
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
        
        