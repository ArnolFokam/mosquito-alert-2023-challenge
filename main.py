import logging
import os
import sys
import hydra
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch

from mosquito.models import models
from mosquito.datasets import datasets
from mosquito.transforms import transforms
from mosquito.helpers import get_dir, get_new_run_dir_params, has_valid_hydra_dir_params, time_activity, log

def train_one_epoch(dataloader, model, optimizers, device, log_every_n_steps, epoch, results_dir):
    model.train()
    
    total_loss, total_num = 0.0, 0
    for i, batch in enumerate(dataloader):
        
        for optimizer in optimizers:
            optimizer.zero_grad()
            
        img, target = batch
        
        # filter out images without annotations and move to device
        keep = set([i for i in range(len(target)) if len(target[i]["boxes"]) > 0])
        img = [img[i].to(device) for i in range(len(img)) if i in keep]
        target = [{k: v.to(device) for k, v in target[i].items()} for i in range(len(target)) if i in keep]

        loss_dict = model(img, target)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        
        total_loss += loss.item()
        total_num += 1
            
        for optimizer in optimizers:
            optimizer.step()
        
        global_step = max(epoch - 1, 0) * len(dataloader) + i + 1
        if global_step % log_every_n_steps == 0:
            logging.info(f"Epoch: {epoch} | Step: {global_step} | Loss: {loss.item()}")
            log(results_dir, {"train-loss": total_loss / total_num}, step=global_step)



def evaluate(dataloader, model, device, results_dir, epoch):
    model.eval()
    
    total_loss = 0
    
    for _, batch in enumerate(dataloader):
            
        img, target = batch
    
        # filter out images without annotations and move to device
        keep = set([i for i in range(len(target)) if len(target[i]["boxes"]) > 0])
        img = [img[i].to(device) for i in range(len(img)) if i in keep]
        target = [{k: v.to(device) for k, v in target[i].items()} for i in range(len(target)) if i in keep]

        with torch.no_grad():
            loss_dict = model(img, target)
            loss = sum(loss for loss in loss_dict.values())
            
        total_loss += loss.item()
    global_step = epoch * len(dataloader)
    log(results_dir, {"train-loss": total_loss / len(dataloader)}, step=global_step)
    logging.info(f"Validation Loss: {total_loss / len(dataloader)}")


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
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
    print(output_dir)
    
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
        
    # create model and optimizers
    model = models[cfg.model_name](cfg, train_dataset.num_classes).to(device)
    optimizers = model.configure_optimizers()
    
    with time_activity("Training"):
        
        for epoch in range(cfg.num_epochs):
            
            with time_activity("Epoch {}".format(epoch + 1)):
                
                # train for one epoch
                train_one_epoch(
                    train_dataloader,
                    model, 
                    optimizers, 
                    device,
                    log_every_n_steps =cfg.log_every_n_steps, 
                    epoch=epoch, 
                    results_dir=output_dir
                )
                    
                # evaluate on the val dataset
                if val_dataloader is not None:
                    evaluate(
                        val_dataloader,
                        model,
                        device,
                        epoch=epoch,
                        results_dir=output_dir
                    )
                    
    # save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    logging.info(f"model saved at {output_dir}")
                
if __name__ == "__main__":
    if has_valid_hydra_dir_params(sys.argv):
        main()
    else:
        params = get_new_run_dir_params()
        for param, value in params.items():
            sys.argv.append(f"{param}={value}")
    main()