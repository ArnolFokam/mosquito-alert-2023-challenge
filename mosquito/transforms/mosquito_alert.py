from typing import Any
import albumentations as A

from mosquito.transforms.base import BaseTransform


class MosquitoAlertTransformv0(BaseTransform):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.transform = A.Compose([
            A.RandomCrop(width=self.cfg.input_size, height=self.cfg.input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    min_area=0.01, 
                                    min_visibility=0.1,
                                    label_fields=['labels'])
        )
    
    def __call__(self, image, bbox, label) -> Any:
        transformed = self.transform(image=image, bboxes=[bbox], labels=[label])
        return transformed["image"], transformed["bboxes"]