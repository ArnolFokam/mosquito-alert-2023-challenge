from typing import Any

import torch
import albumentations as A
import albumentations.pytorch
from mosquito.transforms.base import BaseTransform


class MosquitoAlertTransformv0(BaseTransform):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.transform = A.Compose([
            A.Resize(width=self.cfg.input_size, height=self.cfg.input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            albumentations.pytorch.ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    min_area=0.001, 
                                    min_visibility=0.05,
                                    label_fields=['labels'])
        )
    
    def __call__(self, image, target) -> Any:
        transformed = self.transform(image=image, bboxes=target["boxes"], labels=target["labels"])
        return transformed["image"].float() / 255., {
            "boxes":  torch.tensor(transformed["bboxes"], dtype=torch.float32), 
            "labels": torch.tensor(transformed["labels"], dtype=torch.int64),
            "image_id": torch.tensor(target["image_id"], dtype=torch.int64),
        }