import torch
import os
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset

class ImageData(Dataset):
    def __init__(self, data_dir, target_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.img_names = os.listdir(data_dir)
        self.targt_names = os.listdir(target_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.data_dir + img_name
        target_name = img_name[0:-4] + '_mask.pt'
        target_path = self.target_dir + target_name
        #print(img_path)
        image = cv.imread(img_path)
        target = torch.load(target_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(target)

        
        return image, target