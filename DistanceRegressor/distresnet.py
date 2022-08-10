# imports
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops
from torchvision.models.resnet import ResNet,Bottleneck
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np
import math
import sys

# define a custom resnext50 backbone
class ResNet18(ResNet):
    def __init__(self):
        kwargs = {}
        kwargs['groups']=32
        kwargs['width_per_group']=4
        super(ResNet18, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        del self.avgpool
        del self.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# mlp class for distance regression given features as input
class DistanceRegressor(nn.Module):
    def __init__(self):
        super(DistanceRegressor, self).__init__()
        
        # define our fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        
        # define activation layers
        self.relu = nn.ReLU(inplace=True)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
    
# mlp class for keypoint regression given features as input
class KeypointRegressor(nn.Module):
    def __init__(self):
        super(KeypointRegressor, self).__init__()
        
        # define our fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)
        
        # define activation layers
        self.relu = nn.ReLU(inplace=True)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

# entire model class for distance regression
class DistResNet18(nn.Module):
    def __init__(self, n_classes, image_size=256, pretrained=True, keypoints=False):
        super(DistResNet18, self).__init__()
        
        # model class params
        self.image_size = image_size
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.keypoints = keypoints
        
        # model backbone
        resnet_net = torchvision.models.resnet18(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        
        # define an ROI pooling layer
        self.RoIPool = ops.RoIPool(output_size=(1,1), spatial_scale=32.0/self.image_size)
        
        # mlp classifier head/layer
        self.classifier = nn.Linear(512, self.n_classes)
        
        # mlp distance regressor head
        self.distance_regressor_head = DistanceRegressor()
        
        # 3d keypoint regressor head
        if self.keypoints:
            self.keypoint_regressor_head = KeypointRegressor()
        
        # load in pretrained resnext50 weights
        # if self.pretrained:
        #     pretrain_state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        #     self.backbone.load_state_dict(pretrain_state_dict, strict=False)
    
    def forward(self, x, x_boxes):
        
        # feed entire image thru the backbone
        x = self.backbone(x)

        # perform roi pooling of resnext features
        x = self.RoIPool(x, x_boxes)
        
        # flatten our feature vector
        x = x.flatten(start_dim=1)

        # feed to classifier
        class_logits = self.classifier(x)
        
        # feed to distance regressor
        distance = self.distance_regressor_head(x)
        
        # if applicable, feed to keypoint regressor
        if self.keypoints:
            keypoints = self.keypoint_regressor_head(x)
            return (class_logits, distance, keypoints)
        else:
            return (class_logits,distance)