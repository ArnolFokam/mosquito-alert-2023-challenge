from typing import Any, Tuple

import torch


class BaseTransform:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def __call__(self, image, bbox) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError