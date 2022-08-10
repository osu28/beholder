# imports
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import transforms as T
from sklearn.model_selection import train_test_split


# dataset class for the nuscenes dataset
class NuScenes(torch.utils.data.Dataset):
    def __init__(self, img_dir, meta_path, split='train', augs=None, transforms=None, size=1024, map_to_kitti=False, cut100=False, norm=True, mlp=False):

        # class attributes
        self.img_dir = img_dir
        self.split = split
        self.augs = augs
        self.transforms = transforms
        self.size = size
        self.mlp = mlp

        # read in metadata
        self.meta_path = meta_path
        self.metadata = pd.read_csv(self.meta_path)

        # cut everything further than 100m
        if cut100:
            self.metadata = self.metadata[self.metadata['distance'] <= 100]

        # normalize distance values between (-1,1)
        self.dist_min = self.metadata['distance'].min()
        self.dist_max = self.metadata['distance'].max()
        if norm:
            self.metadata['distance'] = self.metadata['distance'].apply(
                lambda x: 2*(x-self.dist_min)/(self.dist_max-self.dist_min) - 1)

        if map_to_kitti:
            self.mapping = {}
            with open('/irad_users/walravenp/beholder-interns/nuscenes_class_mapping.txt') as f:
                for line in f:
                    (key, val) = line.split()
                    self.mapping[key] = val
            self.metadata['detection class'] = self.metadata['detection class'].apply(
                lambda x: self.mapping[x])

            # integer encode the class labels
            self.class_to_int_mapping = {}
            with open('/irad_users/walravenp/beholder-interns/label_to_int.txt') as f:
                for line in f:
                    (key, val) = line.split()
                    self.class_to_int_mapping[key] = int(val)
            self.metadata['detection class'] = self.metadata['detection class'].apply(
                lambda x: self.class_to_int_mapping[x])
        # else:
        #     # one-hot encode the class labels
        #     self.classes = np.array(self.metadata['detection class'].unique().tolist())
        #     self.n_classes = len(self.classes)
        #     self.metadata['detection class'] = self.metadata['detection class'].apply(lambda x: np.where(self.classes==x)[0][0])

        # perform norm on the classes
        if self.mlp:
            # no van, no person_sitting
            self.class_df = pd.get_dummies(
                self.metadata['detection class'], prefix='class')
            self.metadata[self.class_df.columns] = self.class_df
            self.metadata.drop('detection class', axis=1, inplace=True)

        # split data at the scene level for train/test
        self.scenes = self.metadata['scene name'].unique()
        _, test = train_test_split(self.scenes, test_size=0.2, random_state=42)
        self.test_scene_idxs = [idx for idx in range(
            len(self.scenes)) if self.scenes[idx] in test]
        self.train_scene_idxs = [idx for idx in range(
            len(self.scenes)) if idx not in self.test_scene_idxs]

        if self.split == 'train':
            self.scenes = self.scenes[self.train_scene_idxs]
            self.metadata = self.metadata[self.metadata['scene name'].isin(
                self.scenes)]
        else:
            self.scenes = self.scenes[self.test_scene_idxs]
            self.metadata = self.metadata[self.metadata['scene name'].isin(
                self.scenes)]

        # index the metadata by image id
        self.metadata.set_index(
            ['sample token', 'image filename'], inplace=True)

        # list of unique image_ids
        self.sample_list = self.metadata.index.unique().tolist()

        # clip bounding boxes to size of the image
        self.metadata['bbox y-max'] = self.metadata['bbox y-max'].apply(
            lambda x: x if x <= 899 else 899)
        self.metadata['bbox x-max'] = self.metadata['bbox x-max'].apply(
            lambda x: x if x <= 1599 else 1599)

        # define our resize transform
        self.resize = T.Resize((self.size, self.size))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        # get the image
        image_file = self.sample_list[idx][1]
        image_path = os.path.join(self.img_dir, image_file)
        image = np.array(Image.open(image_path))

        # define our targets data to grab
        data = self.metadata.loc[self.sample_list[idx]]

        # get the boxes, distances and classes
        boxes = np.array([data['bbox x-min'].tolist(),
                          data['bbox y-min'].tolist(),
                          data['bbox x-max'].tolist(),
                          data['bbox y-max'].tolist()
                          ])
        boxes = np.reshape(np.transpose(boxes), (len(data), 4))
        distances = np.array(data['distance'])

        if self.mlp:
            classes = np.array(data[self.class_df.columns])
        else:
            classes = np.array(data['detection class'].tolist())

        if self.augs:
            transformed = self.augs(image=image, bboxes=boxes)
            image, boxes = transformed['image'], np.array(
                transformed['bboxes'])

        if self.transforms:
            image, boxes = self.resize(image, boxes)
            image = self.transforms(image)

            if self.mlp:
                boxes = boxes / self.size

        return (image,
                torch.tensor(boxes).type(torch.FloatTensor),
                torch.tensor(distances).type(torch.FloatTensor),
                torch.tensor(classes).type(torch.LongTensor))

    # custom collate function since each batch will have different number of objects
    def collate_fn(self, batch):

        images = list()
        boxes = list()
        distances = list()
        classes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            distances.append(b[2])
            classes.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, distances, classes


# dataset for the KITTI dataset
class KITTI(torch.utils.data.Dataset):
    def __init__(self, img_dir, meta_path, split='train', augs=None, transforms=None, size=1024, norm=True):

        # class attributes
        self.img_dir = img_dir
        self.split = split
        self.augs = augs
        self.transforms = transforms
        self.size = size

        # read in metadata
        self.meta_path = meta_path
        self.metadata = pd.read_csv(self.meta_path)

        # remove classes we don't care about from the dataframe
        self.metadata = self.metadata.loc[self.metadata['class'] != 'DontCare']

        # remove objects that are occluded
        self.metadata = self.metadata.loc[self.metadata['occluded'] <= 1]

        # index by file name
        self.metadata.set_index(['filename'], inplace=True)

        # compute distances from 3D coordinates in dataframe
        self.metadata['distance'] = self.metadata.apply(lambda x: (
            x['xloc']**2 + x['yloc']**2 + x['zloc']**2)**0.5, axis=1)

        # normalize distance values between (-1,1)
        self.dist_min = self.metadata['distance'].min()
        self.dist_max = self.metadata['distance'].max()
        if norm:
            self.metadata['distance'] = self.metadata['distance'].apply(
                lambda x: 2*(x-self.dist_min)/(self.dist_max-self.dist_min) - 1)

        # integer encode the class labels
        self.class_to_int_mapping = {}
        with open('/irad_users/walravenp/beholder-interns/label_to_int.txt') as f:
            for line in f:
                (key, val) = line.split()
                self.class_to_int_mapping[key] = int(val)
        self.metadata['class'] = self.metadata['class'].apply(
            lambda x: self.class_to_int_mapping[x])

        # list of unique image_ids
        self.sample_list = self.metadata.index.unique().tolist()

        # sample dataset
        train, test = train_test_split(
            self.sample_list, test_size=0.2, random_state=42)
        self.train_samples = [self.sample_list[i] for i in range(
            len(self.sample_list)) if self.sample_list[i] in train]
        self.test_samples = [self.sample_list[i] for i in range(
            len(self.sample_list)) if self.sample_list[i] in test]

        assert not (set(self.train_samples) & set(self.test_samples))

        if self.split == 'train':
            self.sample_list = self.train_samples
            self.metadata.drop(index=self.test_samples, inplace=True)
        else:
            self.sample_list = self.test_samples
            self.metadata.drop(index=self.train_samples, inplace=True)

        # define our resize transform
        self.resize = T.Resize((self.size, self.size))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        # get the image
        image_path = os.path.join(
            self.img_dir, self.sample_list[idx].split('.txt')[0]+'.png')
        image = np.array(Image.open(image_path))

        # define our tragets data to grab
        data = self.metadata.loc[self.sample_list[idx]]

        # get the boxes, distances and classes
        boxes = np.array([data['xmin'].tolist(),
                          data['ymin'].tolist(),
                          data['xmax'].tolist(),
                          data['ymax'].tolist()
                          ])

        distances = np.expand_dims(np.array(data['distance'].tolist()), axis=0) if len(
            boxes.shape) == 1 else np.array(data['distance'])
        classes = np.expand_dims(np.array(data['class'].tolist()), axis=0) if len(
            boxes.shape) == 1 else np.array(data['class'])
        boxes = np.expand_dims(boxes, axis=0) if len(
            boxes.shape) == 1 else np.reshape(np.transpose(boxes), (len(data), 4))

        if self.augs:
            transformed = self.augs(image=image, bboxes=boxes)
            image, boxes = transformed['image'], np.array(
                transformed['bboxes'])

        if self.transforms:
            image, boxes = self.resize(image, boxes)
            image = self.transforms(image)

        sample = [image,
                  torch.tensor(boxes).type(torch.FloatTensor),
                  torch.tensor(distances).type(torch.FloatTensor),
                  torch.tensor(classes).type(torch.LongTensor)]

        return sample

    # custom collate function since each batch will have different number of objects
    def collate_fn(self, batch):

        images = list()
        boxes = list()
        distances = list()
        classes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            distances.append(b[2])
            classes.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, distances, classes
