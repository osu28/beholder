# imports
import torch
import torchvision
import os
import pandas as pd
import numpy as np
from PIL import Image
import transforms as T
from sklearn.model_selection import train_test_split

# dataset class for the intern custom dataset
class InternData(torch.utils.data.Dataset):
    def __init__(self, img_dir, meta_path, split='train', augs=None, transforms=None, size=1024, map_to_nuscenes=False, map_to_kitti=False, norm=True):
        
        # class attributes
        self.img_dir = img_dir
        self.split = split
        self.augs = augs
        self.transforms = transforms
        self.size = size
        
        # read in metadata
        self.meta_path = meta_path
        self.metadata = pd.read_csv(self.meta_path)

        # remove negative distance values
        self.metadata = self.metadata[self.metadata['distance'] >= 0]

        # randomize rows
        # self.metadata = self.metadata.sample(frac=1)

        # normalize distance values between (-1,1)
        # self.dist_min = self.metadata['distance'].min()
        # self.dist_max = self.metadata['distance'].max()
        self.dist_min = 0.914273901974966
        self.dist_max = 311.4842790022132
        if norm:
            self.metadata['distance'] = self.metadata['distance'].apply(lambda x: 2*(x-self.dist_min)/(self.dist_max-self.dist_min) -1)

        if map_to_nuscenes:
            self.metadata['class'] = self.metadata['class'].apply(lambda x: 2 if x == 'car' else 3)  # maps to car (2) or person (3)
        elif map_to_kitti:
            self.metadata['class'] = self.metadata['class'].apply(lambda x: 2 if x == 'car' else 0)  # maps to car (2) or pedestrian (0)  
        else:
            # one-hot encode the class labels
            self.classes = np.array(self.metadata['class'].unique().tolist())
            self.n_classes = len(self.classes)
            self.metadata['class'] = self.metadata['class'].apply(lambda x: np.where(self.classes==x)[0][0])

        # index the metadata by image id
        self.metadata.set_index(['name'], inplace=True)

        # list of unique image_ids
        self.sample_list = self.metadata.index.unique().tolist()

        # sample dataset
        self.test_samples = [
            # oscar samples
            '000183.jpg',
            '000184.jpg',
            '000185.jpg',
            '000186.jpg',
            '000187.jpg',
            '000188.jpg',
            '000189.jpg',
            '000190.jpg',
            '000191.jpg',
            '000192.jpg',
            '000193.jpg',
            '000194.jpg',
            '000195.jpg',
            '000196.jpg',
            '000197.jpg',
            '000198.jpg',
            '000199.jpg',
            '000200.jpg',
            '000201.jpg',
            '000202.jpg',
            '000203.jpg',
            '000204.jpg',
            '000205.jpg',
            '000206.jpg',
            '000207.jpg',
            '000208.jpg',
            '000209.jpg',
            # kaitlin and preston samples
            '000244.jpg',
            '000248.jpg',  # 
            '000249.jpg',
            '000261.jpg',  # 
            '000262.jpg',
            '000264.jpg',  # 
            '000265.jpg',
            '000266.jpg',  # 
            '000267.jpg',
            '000272.jpg',  # 
            '000273.jpg',
            '000290.jpg',  # 
            '000291.jpg',
            '000292.jpg',  # 
            '000299.jpg',
            '000300.jpg',  # 
            '000309.jpg',
            '000310.jpg',  # 
            '000311.jpg',
            '000312.jpg',  # 
            '000313.jpg',
            '000315.jpg',  # 
            '000327.jpg',
            '000328.jpg',  # 
            '000329.jpg',  # 
            '000330.jpg',  # 
            '000333.jpg',  # 
            '000334.jpg',  # 
            '000335.jpg',  # 
            '000336.jpg',  # 
            '000356.jpg',  # 
            '000357.jpg',  # 
            '000358.jpg',  # 
            '000372.jpg',  # 
            '000387.jpg',  # 
            '000388.jpg',  # 
        ]

        # self.remaining = self.sample_list.copy()

        # for item in self.untouched_samples:
        #     self.remaining.remove(item)

        # train, test = train_test_split(self.remaining, test_size=0.2, train_size=0.8, random_state=42)
        # self.train_samples = [self.sample_list[i] for i in range(len(self.sample_list)) if self.sample_list[i] in train]
        # self.test_samples = [self.sample_list[i] for i in range(len(self.sample_list)) if self.sample_list[i] in test]

        # self.test_samples = [self.sample_list[i] for i in range(len(self.sample_list)) if self.sample_list[i] in self.untouched_samples]
        self.train_samples = [self.sample_list[i] for i in range(len(self.sample_list)) if self.sample_list[i] not in self.test_samples]

        # assert not (set(self.untouched_samples) & set(self.train_samples))
        # assert not (set(self.untouched_samples) & set(self.test_samples))
        assert not (set(self.test_samples) & set(self.train_samples))
        
        if self.split=='train':
            self.sample_list = self.train_samples
            self.metadata.drop(index=self.test_samples, inplace=True)
            # self.metadata.drop(index=self.untouched_samples, inplace=True)
        # elif self.split=='untouched':
        #     self.sample_list = self.untouched_samples
        #     self.metadata.drop(index=self.train_samples, inplace=True)
        #     self.metadata.drop(index=self.test_samples, inplace=True)
        else:
            self.sample_list = self.test_samples
            self.metadata.drop(index=self.train_samples, inplace=True)
            # self.metadata.drop(index=self.untouched_samples, inplace=True)

        # define our resize transform
        self.resize = T.Resize((self.size, self.size))
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        
        # get the image
        image_file = self.sample_list[idx]
        image_path = os.path.join(self.img_dir, image_file)
        image = np.array(Image.open(image_path))
        
        # define our targets data to grab
        data = self.metadata.loc[[self.sample_list[idx]]]
        
        # get the boxes, distances and classes
        boxes = np.array([data['x-min'].tolist(),
                          data['y-min'].tolist(),
                          data['x-max'].tolist(),
                          data['y-max'].tolist()
                         ])

        boxes = np.reshape(np.transpose(boxes), (len(data), 4))
        distances = np.array(data['distance'])
        classes = np.array(data['class'].tolist())
        
        if self.augs:
            transformed = self.augs(image=image, bboxes=boxes)
            image,boxes = transformed['image'],np.array(transformed['bboxes'])
        
        if self.transforms:
            image,boxes = self.resize(image,boxes)
            image = self.transforms(image)
        
        return (image,torch.tensor(boxes).type(torch.FloatTensor),
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
