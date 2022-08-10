# make custom scripts visible to python path
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import metrics
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
import itertools
import torch
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from inference.feature_extractor import FeatureExtractor
from inference.mlp import DistanceRegressor
from Datasets.dataset import NuScenes
from albumentations import (OneOf, 
                            HueSaturationValue,
                            RandomBrightnessContrast,
                            Blur,
                            GaussNoise,
                            CLAHE,
                            RGBShift)


class FeatureFusionTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):

        # basic stuff
        torch.manual_seed(42)
        np.random.seed(42)
        self.context = context
        self.image_size = 1024
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_fvs = True
        self.n_components = 12

        # loss hparams
        self.delta = self.context.get_hparam('delta')
        self.dist_criterion = torch.nn.HuberLoss(delta=self.delta)
        self.key_criterion = torch.nn.MSELoss()
        self.lambda_d = self.context.get_hparam('lambda_d')

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
        self.feature_extractor = FeatureExtractor(8, self.image_size, False, True)
        # trained on Nuscenes, finetuned on InternData both mapped to KITTI
        self.feature_extractor.load_state_dict(torch.load('/irad_users/determined/checkpoints/10cd963a-b8b5-4e0a-b02a-db518963877c/state_dict.pth')['models_state_dict'][0], strict=True)
        self.feature_extractor.to(self.device)

        self.pca = PCA()
        # TODO add this in

        model = DistanceRegressor(n_features=33)
        model.to(self.device)
        return self.context.wrap_model(model)

    def train_batch(self,
                    batch,
                    epoch_idx: int,
                    batch_idx: int):

        # bring data to gpu
        inputs, targets = batch[0].to(self.device).float(), batch[1].to(self.device).float()

        # zero param gradients
        self.optimizer.zero_grad()

        # forward pass
        preds = self.model(inputs)

        # loss and backprop
        dist_loss = self.dist_criterion(preds, targets.unsqueeze(1))

        # backprop and step optimizer
        self.context.backward(dist_loss)
        self.context.step_optimizer(self.optimizer)

        return {'distance_loss': dist_loss}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.dataloader.DataLoader):

        # lists to hold outputs
        targets_list = []
        preds_list = []
        test_loss = 0

        # get our model results
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader, 0):

                inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
                preds = self.model(inputs)
                test_loss += self.dist_criterion(preds, targets.unsqueeze(1)).item()

                targets_list.append(targets.cpu().numpy())
                preds_list.append(preds.flatten().cpu().numpy())

        # flatten our list of lists
        distance_targets_arr = list(
            itertools.chain.from_iterable(targets_list))
        distance_pred_list = list(itertools.chain.from_iterable(preds_list))

        # bring distance predictions and gts to flattened arrays
        distance_targets_arr = np.concatenate([array.flatten() for array in distance_targets_arr])
        distance_pred_list = np.concatenate([array.flatten() for array in distance_pred_list])

        # unnormalize distances
        distance_targets_arr = (distance_targets_arr + 1) * (self.test_dataset.dist_max - self.test_dataset.dist_min) / 2.0 + self.test_dataset.dist_min
        distance_pred_list = (distance_pred_list + 1) * (self.test_dataset.dist_max - self.test_dataset.dist_min) / 2.0 + self.test_dataset.dist_min

        # calculate metrics
        abs_rel_dist = np.mean(metrics.abs_relative_distance(distance_pred_list, distance_targets_arr))
        sq_rel_dist = metrics.sq_relative_distance(distance_pred_list, distance_targets_arr)
        rmse = metrics.rmse(distance_pred_list, distance_targets_arr)
        log_rmse = metrics.log_rmse(distance_pred_list, distance_targets_arr)

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
        self.train_data = NuScenes(img_dir='/irad_mounts/lambda-quad-5-data/beholder/intern_data/nuscenes-full/',
                                      meta_path='/irad_mounts/lambda-quad-5-data/beholder/intern_data/nuscenes-full/nuscenes-v1.0.csv',
                                      split = 'train',
                                      augs = train_aug,
                                      transforms = self.transforms,
                                      size = self.image_size,
                                      map_to_kitti=True
                                      )

        trainloader = DataLoader(
            self.train_dataset, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, drop_last=False)

        return trainloader

    def build_validation_data_loader(self):
        # self.test_dataset = NuScenesDataset(
        #     self.data, split='test', fvs=self.use_fvs, n_components=self.n_components)

        testloader = DataLoader(
            self.test_dataset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, drop_last=False)

        return testloader
