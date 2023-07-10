import logging
import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
from mosquito.datasets import datasets

from mosquito.helpers import get_dir


@hydra.main(version_base=None, config_path=None)
def train(cfg: DictConfig):
    """Main training script"""
    
    # ensure reprodcibility 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    else:
        raise Exception("Code can only run on a value GPU")
        
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # speed up
    torch.backends.cudnn.benchmark = True
    
    # initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # initializing results dir
    output_dir = get_dir(HydraConfig.get().runtime.output_dir)
    
    # save configuration used at the folder location
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)
        
    # get training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get train and val datasets
    train_dataset, val_dataset = datasets[cfg.dataset_name].get_train_and_val_dataset(cfg)

if __name__ == "__main__":
    train()