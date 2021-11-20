from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
import os
import torch
import albumentations as A 
from albumentations.augmentations.transforms import *
from torchvision.io import read_image
from torchvision.transforms import RandomCrop
import numpy as np
from utils import *


class Xray_Dataset(Dataset):
    def __init__(self, root_dir, extension = 'jpg', mode = 'ChestXpert'):
        if mode == 'ChestXpert':
            self.data_list = glob(os.path.join(root_dir,'*/*/*/*.'+extension))
        else:
            self.data_list = glob(os.path.join(root_dir,'*.'+extension))
        self.blur_transform = A.Compose([
            A.OneOf([
                MotionBlur(p =0.7, blur_limit = (5,30))
                ,Blur(p =0.3, blur_limit = [5,30])
                ])
        ])
        self.crop = RandomCrop( size = (1800, 2100), pad_if_needed = True)
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img = read_image(self.data_list[idx]).float()
        img = self.crop(img)
        target = img / 255.0
        img = img / 255.0
        if np.random.uniform() > 0.5:
            img = self.blur_transform(image = img.numpy().squeeze(0))['image']
            img = torch.from_numpy(img).unsqueeze(0)
        else:
            img = sinc_filter(torch.tensor(img))
            img = img.squeeze(0).numpy()
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
    
        if np.random.uniform() > 0.5:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bilinear', recompute_scale_factor=True, align_corners = True).squeeze(0)
        else:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bicubic', recompute_scale_factor=True, align_corners = True).squeeze(0)
        if np.random.uniform() > 0.5:
            img = random_add_poisson_noise(img, scale_range = [0.05, 3], clip = False, rounds = False)
        else:
            img = random_add_gaussian_noise(img, sigma_range = [1, 30], clip = False, rounds = False)

        img = self.blur_transform(image = img.numpy().squeeze(0))['image']
        img = torch.from_numpy(img).unsqueeze(0)

        if np.random.uniform() > 0.5:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bilinear', recompute_scale_factor=True, align_corners = True).squeeze(0)
        else:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bicubic', recompute_scale_factor=True, align_corners = True).squeeze(0)

        if np.random.uniform() > 0.5:
            img = random_add_poisson_noise(img, scale_range = [0.05, 2.5], clip = False, rounds = False)
        else:
            img = random_add_gaussian_noise(img, sigma_range = [1, 25], clip = False, rounds = False)
        
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        img = sinc_filter(img)
        img = img.squeeze(0)
        return img, target

