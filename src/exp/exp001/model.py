from torch import nn
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Just a dummy value for the classification head
NUM_CLASSES = 2

def get_model(box_detections_per_img=600, pretrained=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained,
                                                               box_detections_per_img=box_detections_per_img,
                                                               image_mean=[0.485, 0.456, 0.406],
                                                               image_std=[0.229, 0.224, 0.225])

    #get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #replace the pre-trainde head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    #get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    return model


if __name__ == "__main__":
    model = get_model(pretrained=False)
    print(model)
