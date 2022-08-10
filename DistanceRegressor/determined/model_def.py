# make custom scripts visible to python path
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

# imports
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools
from typing import cast, Tuple, Dict
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from albumentations import OneOf, Compose
from albumentations import (HorizontalFlip,
                            HueSaturationValue,
                            RandomBrightnessContrast,
                            Blur,
                            GaussNoise,
                            CLAHE,
                            CoarseDropout,
                            RGBShift,
                            BboxParams)

# custom imports
from distnet import DistResNeXt50
from Datasets.dataset import NuScenes, KITTI
import metrics

class DistNetTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        
        # basic stuff
        self.context = context
        self.image_size = 1024
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # loss hparams
        self.class_criterion = torch.nn.CrossEntropyLoss()
        self.dist_criterion = torch.nn.HuberLoss()
        self.key_criterion = torch.nn.MSELoss()
        self.lambda_d = self.context.get_hparam('lambda_d')
        self.lambda_k = self.context.get_hparam('lambda_k')
        
        # model and optimizer build
        self.model = self.build_model()
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(
                self.model.parameters(),
                weight_decay=self.context.get_hparam('weight_decay'),
                lr=self.context.get_hparam('learning_rate')
            )
        )
        
    def build_model(self):
        model = DistResNeXt50(n_classes=8, image_size=self.image_size, keypoints=True)
        model.to(self.device)
        return self.context.wrap_model(model)

    def train_batch(self,
                    batch,
                    epoch_idx: int,
                    batch_idx: int):
        
        # grab the batch and move the batch to the gpu
        inputs,boxes,distances,classes = batch[0],batch[1],batch[2],batch[3]
        inputs = inputs.to(self.device)
        boxes = [b.to(self.device) for b in boxes]
        distances = torch.cat([d.to(self.device) for d in distances])
        classes = torch.cat([c.to(self.device) for c in classes])
        
        # zero param gradients
        self.optimizer.zero_grad()
        
        # forward pass
        class_preds,distance_preds,keypoint_preds = self.model(inputs, boxes)
        
        # loss calculation
        class_loss = self.class_criterion(class_preds.squeeze(), classes)
        dist_loss = self.lambda_d*self.dist_criterion(distance_preds.squeeze(), distances)
        key_loss = self.lambda_k*self.key_criterion(keypoint_preds, torch.cat(boxes)/self.image_size)
        loss = class_loss + dist_loss + key_loss
        
        # backprop and step optimizer
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        
        return {'total_loss': loss,
                'class_loss': class_loss,
                'distance_loss': dist_loss,
                'keypoint_loss': key_loss}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.dataloader.DataLoader):
        
        # lists to hold outputs
        distance_pred_list = []
        distance_gt_list = []
        
        # get our model results
        self.model.eval()
        with torch.no_grad():
            for i,data in enumerate(data_loader, 0):

                # grab the batch and move the batch to the gpu
                inputs,boxes,distances,classes = data[0],data[1],data[2],data[3]
                inputs = inputs.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                distances = torch.cat([d.to(self.device) for d in distances])
                classes = torch.cat([c.to(self.device) for c in classes])
                
                # save the len of the bbox matrix to organize samples by
                num_obj = [len(bbox_matrix) for bbox_matrix in boxes]

                # forward, backward, optimize
                _,distance_preds,_ = self.model(inputs, boxes)
                distance_preds = distance_preds.cpu().numpy()
                distance_gts = distances.cpu().numpy()
                
                # reshape arrays to be batch x n_objects
                distance_preds = [distance_preds[i:i+num_obj[i]] for i in range(len(num_obj))]
                distance_gts = [distance_gts[i:i+num_obj[i]] for i in range(len(num_obj))]
                
                # append to list
                distance_pred_list.append(distance_preds)
                distance_gt_list.append(distance_gts)
                
        # flatten our list of lists
        distance_gt_list = list(itertools.chain.from_iterable(distance_gt_list))
        distance_pred_list = list(itertools.chain.from_iterable(distance_pred_list))
                
        # bring distance predictions and gts to flattened arrays
        distance_preds = np.concatenate([array.flatten() for array in distance_pred_list])
        distance_gts = np.concatenate([array.flatten() for array in distance_gt_list])

        # unnormalize distances
        distance_preds = (distance_preds+1)*(self.test_data.dist_max-self.test_data.dist_min)/2.0 + self.test_data.dist_min
        distance_gts = (distance_gts+1)*(self.test_data.dist_max-self.test_data.dist_min)/2.0 + self.test_data.dist_min

        # calculate metrics
        abs_rel_dist = np.mean(metrics.abs_relative_distance(distance_preds, distance_gts))
        sq_rel_dist = metrics.sq_relative_distance(distance_preds, distance_gts)
        rmse = metrics.rmse(distance_preds, distance_gts)
        log_rmse = metrics.log_rmse(distance_preds, distance_gts)

        return {'Abs Rel': abs_rel_dist,
                'Sq Rel': sq_rel_dist,
                'RMSE': rmse,
                'RMSE log': log_rmse}

    def build_training_data_loader(self):

        # define an albumentations augmentation routine
        alb_aug_list = [HueSaturationValue(hue_shift_limit=20,
                                           sat_shift_limit=30,
                                           val_shift_limit=20,
                                           p=1),
                        RandomBrightnessContrast(brightness_limit=0.2,
                                                 contrast_limit=0.2,
                                                 brightness_by_max=True,
                                                 p=1), 
                        Blur(blur_limit=7,
                             p=1),
                        GaussNoise(var_limit=(10.0, 50.0),
                                   mean=0,
                                   p=1),
                        CLAHE(clip_limit=4.0,
                              tile_grid_size=(8, 8),
                              p=1),
                        RGBShift(p=1)
                        ]
        train_aug = OneOf(alb_aug_list, p=0.7)
        
        # create a transforms routine
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]
                                                          )])
        
        # create training dataset and loader
        self.kitti_data = KITTI(img_dir='/irad_mounts/lambda-quad-5-data/beholder/KITTI-distance-estimation/original_data/train_images/',
                                meta_path='/irad_mounts/lambda-quad-5-data/beholder/KITTI-distance-estimation/annotations.csv',
                                split = 'train',
                                augs = train_aug,
                                transforms=self.transforms,
                                size=self.image_size)
        self.nuscenes_data = NuScenes(img_dir='/irad_mounts/lambda-quad-5-data/beholder/nuscenes/',
                                      meta_path='/irad_mounts/lambda-quad-5-data/beholder/nuscenes/nuscenes-v1.0-mini.csv',
                                      split = 'train',
                                      augs = train_aug,
                                      transforms = self.transforms,
                                      size = self.image_size,
                                      map_to_kitti = True)
        
        self.train_data = torch.utils.data.ConcatDataset([self.kitti_data, self.nuscenes_data])
        trainloader = DataLoader(self.train_data,
                                 collate_fn=self.kitti_data.collate_fn,
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 shuffle=True,
                                 drop_last=True)
        return trainloader

    def build_validation_data_loader(self):
        
        self.test_data = NuScenes(img_dir='/irad_mounts/lambda-quad-5-data/beholder/nuscenes/',
                                  meta_path='/irad_mounts/lambda-quad-5-data/beholder/nuscenes/nuscenes-v1.0-mini.csv',
                                  split = 'test',
                                  augs = None,
                                  transforms = self.transforms,
                                  size = self.image_size,
                                  map_to_kitti = True)

        testloader = DataLoader(self.test_data,
                                batch_size=self.context.get_per_slot_batch_size(),
                                drop_last=True,
                                shuffle=False,
                                collate_fn=self.test_data.collate_fn)
        return testloader
