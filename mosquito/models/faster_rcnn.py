import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from mosquito.models.base import BaseModel



class FasterRCNN(BaseModel):
    def __init__(self, cfg, num_classes) -> None:
        super().__init__(cfg)
        
        # load Faster RCNN pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # get the number of input features 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # define a new head for the detector with required  of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, img, target = None):
        return self.model(img, target)
    
    def postprocess(self, outputs):
        for image_id, output in outputs.items():
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]
            
            # filter out the predictions with low scores
            keep = scores > self.cfg.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # filter out the predictions with low scores
            keep = torchvision.ops.nms(boxes, scores, self.cfg.iou_threshold)
            
            if keep.shape[0] == 0:
                outputs[image_id]['boxes'] = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
                outputs[image_id]['scores'] = torch.tensor([1.0])
                outputs[image_id]['labels'] = torch.tensor(self.cfg.default_label_index)
            else:
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # get the mean of the boxes
                boxes = boxes.mean(dim=0, keepdim=True)
                
                # set the label from the box with the highest score
                labels = labels[torch.argmax(scores)]
                
                outputs[image_id]["boxes"] = boxes
                outputs[image_id]["scores"] = scores
                outputs[image_id]["labels"] = labels
                
        return outputs
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    momentum=self.cfg.momentum,
                                    weight_decay=self.cfg.weight_decay)
        return [optimizer]