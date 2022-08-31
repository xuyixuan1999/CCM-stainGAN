import glob
import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, batch_size=None,):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.batch_size = batch_size

        self.files_A = sorted(glob.glob(os.path.join(root, 'train/A') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'train/B') + '/*.png'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ClsDataset(Dataset):
    '''
    The image's name of class dataset must add category label in the end of name , which split of '_'.
    Such as: img_001_0.png, img_002_0.png, img_003_1.png, img_004_1.png. 
    The last number is label of the dataset.
    '''
    def __init__(self, root, transforms_=None, unaligned=False, batch_size=None):
        self.transforms = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.batch_size = batch_size
        self.files_A = sorted(glob.glob(os.path.join(root, 'train/cls_A') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'train/cls_B') + '/*.png'))
        
    def __getitem__(self, index):
        name_A = self.files_A[index % len(self.files_A)]   
         
        if self.unaligned:
            name_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            name_B = self.files_B[index % len(self.files_B)]
            
        label_A = name_A.split('/')[-1].split('.')[0].split('_')[-1]
        label_B = name_B.split('/')[-1].split('.')[0].split('_')[-1]
        label_A = torch.tensor(int(label_A), dtype=torch.long)
        label_B = torch.tensor(int(label_B), dtype=torch.long)
        
        item_A = self.transforms(Image.open(name_A))
        item_B = self.transforms(Image.open(name_B)) 
           
        return {'img_A':item_A, 'img_B':item_B, 'label_A': label_A, 'label_B':label_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDataset(Dataset):
    def __init__(self, root, transforms_=None,):
        self.root = root
        self.transforms = transforms.Compose(transforms_)
        self.files = sorted(os.listdir(root))
        
    def __getitem__(self, index):
        name = self.files[index % len(self.files)]
        path = os.path.join(self.root, name)
        img = self.transforms(Image.open(path))

        return {'img': img, 'name': name}

    def __len__(self):
        return len(self.files)
        
