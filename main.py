import logging
import os
import sys
import hydra
import random
import numpy as np
from sklearn.metrics import f1_score
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch

from mosquito.models import models
from mosquito.datasets import datasets
from mosquito.transforms import transforms
from mosquito.helpers import get_dir, get_new_run_dir_params, has_valid_hydra_dir_params, time_activity, log

def calculate_iou(boxA, boxB):
    # Calculate the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the union area
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union_area = boxA_area + boxB_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    return iou


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


def evaluate(dataloader, model, device, results_dir, epoch, global_step):
    model.eval()
    
    pred_boxes = []
    true_boxes = []
    pred_labels = []
    true_labels = []
    
    for _, batch in enumerate(dataloader):
            
        img, target = batch
    
        # filter out images without annotations and move to device
        keep = set([i for i in range(len(target)) if len(target[i]["boxes"]) > 0])
        img = [img[i].to(device) for i in range(len(img)) if i in keep]
        
        targets = [{k: v.to(device) for k, v in target[i].items()} for i in range(len(target)) if i in keep]

        with torch.no_grad():
            outputs = model(img)
            outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
            outputs = {target["image_id"]: output for target, output in zip(targets, outputs)}
            
            # use the model's postprocess function to get the final predictions
            res = model.postprocess(outputs)
            
            # format targets accordingly
            targets = {target["image_id"]: target for target in targets}
            
            # collection predictions and ground truth
            for image_id in res.keys():
                # append boxes
                pred_boxes.append(res[image_id]["boxes"][0].numpy())
                true_boxes.append(targets[image_id]["boxes"][0].cpu().numpy())
                
                # append labels
                pred_labels.append(res[image_id]["labels"].item())
                true_labels.append(targets[image_id]["labels"].item())
            
    
    macro_f1_score = f1_score(pred_labels, true_labels, average='macro')
    
    iou_scores = []
    for i in range(len(pred_boxes)):
        iou_scores.append(calculate_iou(pred_boxes[i], true_boxes[i]))
        
    mean_iou_score = np.mean(iou_scores)
    
    # log results
    log(results_dir, {"val-macro-f1-score": macro_f1_score, "val-mean-iou-score": mean_iou_score}, step=global_step)
    logging.info(f"Epoch: {epoch} | Step: {global_step} | Macro F1 Score: {macro_f1_score} | Mean IoU Score: {mean_iou_score}")
        
            


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
    
    # get the number of samples per class
    class_counts = np.zeros(train_dataset.dataset.num_classes)
    for i in range(len(train_dataset.indices)):
        _, _, label = train_dataset.dataset.data[i]
        class_counts[label] += 1
    
    # get the class weights
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights)
    
    # get the weights for each sample
    samples_weight = np.array([class_weights[train_dataset.dataset.data[i][2]] for i in range(len(train_dataset.indices))])
    samples_weight = torch.from_numpy(samples_weight)
    
    # create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train_batch_size,
        num_workers=8,
        sampler=torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight)),
        collate_fn=datasets[cfg.dataset_name].collate_fn
    )
    
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=cfg.eval_batch_size, 
            shuffle=False, 
            num_workers=8,
            collate_fn=datasets[cfg.dataset_name].collate_fn
        )
        
    # create model and optimizers
    num_classes = train_dataset.dataset.num_classes if hasattr(train_dataset, "dataset") else train_dataset.num_classes
    model = models[cfg.model_name](cfg, num_classes).to(device)
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
                    epoch=epoch + 1, 
                    results_dir=output_dir
                )
                    
                # evaluate on the val dataset
                if val_dataloader is not None:
                    evaluate(
                        val_dataloader,
                        model,
                        device,
                        epoch=epoch + 1,
                        global_step=(epoch + 1) * len(train_dataloader),
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