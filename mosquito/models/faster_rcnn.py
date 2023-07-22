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
    
    def forward(self, img, target):
        return self.model(img, target)
    
    def configure_optimizers(self):
        # TODO: add support for learning rate schedulers
        
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    momentum=self.cfg.momentum,
                                    weight_decay=self.cfg.weight_decay)
        return [optimizer]