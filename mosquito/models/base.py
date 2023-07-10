import torch

class BaseModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
    def forward(self, x):
        raise NotImplementedError
    
    def configure_optimizers(self):
        raise NotImplementedError