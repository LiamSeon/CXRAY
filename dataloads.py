from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
import os
import albumentations as A 
from albumentations.augmentations.transforms import *
from torchvision.io import read_image
from utils import *
import numpy as np



class Xray_Dataset(Dataset):
    def __init__(self, root_dir):
        self.data_list = glob(os.path.join(root_dir,'*/*/*/*.jpg'))
        self.blur_transform = A.Compose([
            A.OneOf([
                MotionBlur(p =0.7, blur_limit = (20,30))
                ,Blur(p =0.3, blur_limit = [20,30])
                ])
        ])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img = read_image(self.data_list[idx])
        img = img.numpy().astype(float)

        if np.random.uniform() > 0.5:
            img = self.blur_transform(image = img)['image']
        else:
            img = sinc_filter(img)
            img = img.squeeze(0).numpy()

        if np.random.uniform() > 0.5:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bilinear', recompute_scale_factor=True, align_corners = True).squeeze(0).numpy()
        else:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bicubic', recompute_scale_factor=True, align_corners = True).squeeze(0).numpy()

        if np.random.uniform() > 0.5:
            img = random_add_poisson_noise(img, clip = False, rounds = False)
        else:
            img = random_add_gaussian_noise(img, clip = False, rounds = False)


        img = self.blur_transform(image = img)['image']

        if np.random.uniform() > 0.5:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bilinear', recompute_scale_factor=True, align_corners = True).squeeze(0).numpy()
        else:
            img = F.interpolate(img.unsqueeze(0), scale_factor = 0.5, mode ='bicubic', recompute_scale_factor=True, align_corners = True).squeeze(0).numpy()

        if np.random.uniform() > 0.5:
            img = random_add_poisson_noise(img, clip = False, rounds = False)
        else:
            img = random_add_gaussian_noise(img, clip = False, rounds = False)
        
        img = sinc_filter(img)
        img = img.squeeze(0).numpy()

        return img    

