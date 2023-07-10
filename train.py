import logging
import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
from mosquito.transforms import transforms
from mosquito.datasets import datasets

from mosquito.helpers import get_dir, time_activity

def train_one_epoch(dataloader, model, optimizer, device, epoch, log_freq=10):
    pass

def evaluate(dataloader, model, device):
    pass


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
    train_dataset, val_dataset = datasets[cfg.dataset_name].get_train_and_val_dataset(
        cfg,
        transform=transforms[cfg.dataset_name](cfg)
    )
    train_dataloader, val_dataloader = None, None
    
    # create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=8,
        collate_fn=datasets[cfg.dataset_name].collate_fn
    )
    
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8,
            collate_fn=datasets[cfg.dataset_name].collate_fn
        )
        
    # create model
    # model = models[cfg.model_name](cfg)
    
    with time_activity("Training"):
        
        for epoch in range(cfg.num_epochs):
            
            with time_activity("Epoch {}".format(epoch + 1)):
                
                # train for one epoch
                for batch in train_dataloader:
                    # train_one_epoch(
                    #     train_dataloader, 
                    #     model, 
                    #     optimizer, 
                    #     device, 
                    #     epoch, 
                    #     log_freq=cfg.log_freq
                    # )
                    break
                    
                # evaluate on the val dataset
                if val_dataloader is not None:
                    # evaluate(
                    #     val_dataloader,
                    #     model,
                    #     device,
                    # )
                    break
                
if __name__ == "__main__":
    train()