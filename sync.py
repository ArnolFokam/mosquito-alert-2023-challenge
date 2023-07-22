import os
import glob
from omegaconf import OmegaConf
import wandb
import pickle

def sync(dir):
    experiments = glob.glob(f"{dir}/**/completed", recursive=True)
    
    for exp in experiments:
        
        # experiment directory
        exp_dir = os.path.dirname(exp)
        
        syc = os.path.join(exp_dir, "synced")
        if os.path.isfile(syc):
            continue
        
        # open the config file
        with open(os.path.join(exp_dir, 'config.yaml'), 'r') as f:
            cfg = OmegaConf.load(f.name)
            
        # set experiment name
        wandb.init(project=cfg.experiment.name, name=os.path.basename(exp_dir))
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        
        # log things
        trains = glob.glob(os.path.join(exp_dir, '*/'))
        
        for tr in trains:
            
            # get prefix
            prefix = tr.split('/')[-2]
            
            # set prefix
            wandb.define_metric(f"trainer/{prefix}-global-step")
            wandb.define_metric(f"{prefix}/*", step_metric=f"trainer/{prefix}-global-step")
            
            # test things
            wandb.define_metric(f'{prefix}/test-top1-acc', summary='max')
            wandb.define_metric(f'{prefix}/test-top5-acc', summary='max')
            
            
            with open(os.path.join(tr, 'logs.pkl'), 'rb') as f:
                logs = pickle.load(f)
                
            for step, values in logs.items():
                
                # update keys
                values = {f"{prefix}/{k}":v for k, v in values.items()}
                wandb.log({
                    **values, 
                    f"trainer/{prefix}-global-step": step
                })
                    
        
        wandb.finish()
         
        # replace the completed file with sync file     
        os.remove(exp)
        open(os.path.join(os.path.dirname(exp), 'synced'), 'w').close()
            
            
            
if __name__ == "__main__":
    sync("results")