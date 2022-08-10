# make custom scripts visible to python path
import sys
sys.path.append('/irad_users/walravenp/beholder-interns/')

# imports
import sys
import numpy as np
import torch
import torchvision
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from albumentations import (OneOf,
                            HueSaturationValue,
                            RandomBrightnessContrast,
                            Blur,
                            GaussNoise,
                            CLAHE,
                            RGBShift)

# custom imports
from mlp_test import DistanceRegressor
from dataset import NuScenes
import metrics

class TestTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        
        # basic stuff
        torch.manual_seed(42)
        np.random.seed(42)
        self.context = context
        self.image_size = 1024
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # loss hparams
        self.criterion = torch.nn.HuberLoss(delta=self.context.get_hparam('delta'))
        
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
        # instantiate model
        model = DistanceRegressor(n_features=10) # 4 bbox, 6 mapped kitti classes
        model.to(self.device)
        return self.context.wrap_model(model)

    def train_batch(self,
                    batch,
                    epoch_idx: int,
                    batch_idx: int):

        inputs = torch.cat([torch.cat([bbox, class_encoding], dim=1) for bbox, class_encoding in zip(batch[1], batch[3])])
        targets = torch.cat(batch[2])

        # bring data to gpu
        inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
        
        # zero gradients
        self.optimizer.zero_grad()
        
        # forward pass
        preds = self.model(inputs)
        
        # loss and backprop
        loss = self.criterion(preds, targets.unsqueeze(1))

        # backprop and step optimizer
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        
        return {'train_loss': float(loss)}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.dataloader.DataLoader):
        
        # lists to hold outputs
        preds_list = []
        targets_list = []
        test_loss = 0
        
        # get our model results
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs = torch.cat([torch.cat([bbox, class_encoding], dim=1) for bbox, class_encoding in zip(data[1], data[3])])
                targets = torch.cat(data[2])
                inputs,targets = inputs.to(self.device).float(),targets.to(self.device).float()
                preds = self.model(inputs)
                test_loss += self.criterion(preds, targets.unsqueeze(1)).item()
                
                targets_list.append(targets.cpu().numpy())
                preds_list.append(preds.flatten().cpu().numpy())
        
        # metrics calc
        targets_arr = np.concatenate(targets_list, axis=-1)
        preds_arr = np.concatenate(preds_list, axis=-1)
        
        targets_arr = (targets_arr+1)*(self.test_data.dist_max-self.test_data.dist_min)/2.0 + self.test_data.dist_min
        preds_arr = (preds_arr+1)*(self.test_data.dist_max-self.test_data.dist_min)/2.0 + self.test_data.dist_min
        
        rmse = metrics.rmse(preds_arr, targets_arr)

        return {
                'RMSE': rmse,
                'val_loss': test_loss / len(data_loader)
                }

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
                                      map_to_kitti=True,
                                      mlp=True
                                      )
        
        trainloader = DataLoader(self.train_data,
                                 collate_fn=self.train_data.collate_fn,
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 shuffle=True,
                                 drop_last=True,)
        return trainloader

    def build_validation_data_loader(self):
        
        self.test_data = NuScenes(img_dir='/irad_mounts/lambda-quad-5-data/beholder/intern_data/nuscenes-full/',
                                  meta_path='/irad_mounts/lambda-quad-5-data/beholder/intern_data/nuscenes-full/nuscenes-v1.0.csv',
                                  split = 'test',
                                  augs = None,
                                  transforms = self.transforms,
                                  size = self.image_size,
                                  map_to_kitti=True,
                                  mlp=True
                                  )

        testloader = DataLoader(self.test_data,
                                batch_size=self.context.get_per_slot_batch_size(),
                                drop_last=True,
                                shuffle=False,
                                collate_fn=self.test_data.collate_fn)
        return testloader