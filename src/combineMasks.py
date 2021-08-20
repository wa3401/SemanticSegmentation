import torch
from torch import random
import torchvision
import cv2 as cv
import os
import fnmatch
import random

img_dir = '/home/williamanderson/Semantic Segmentation/Final Training Images/finalImages/'
mask_dir = '/home/williamanderson/Semantic Segmentation/Final Training Images/finalMasks/'
target_dir = '/home/williamanderson/Semantic Segmentation/images/willTargets/'
resize_dir = '/home/williamanderson/Semantic Segmentation/images/willResize/'

IM_HEIGHT = 256
IM_WIDTH = 256
NUM_CHANNELS = 4

img_file_list = os.listdir(img_dir)
mask_file_list = os.listdir(mask_dir)
# print(file_list)
jpg_pattern = '*.jpg'
car_pattern = '_class_F1Tenth Car.png'
line_pattern = '_class_Lane Line.png'
lane_pattern = '_class_Current Lane.png'
back_pattern = '_class_Background.png'


for filename in img_file_list:
    if filename != '.DS_Store':
        print(filename)
        target_tensor = torch.zeros((NUM_CHANNELS, 256, 256))
        print(target_tensor.shape)
        if fnmatch.fnmatch(filename, jpg_pattern):
            origPath = img_dir + filename
            orig_img = cv.imread(origPath)
            resize_img = cv.resize(orig_img, (256, 256))
            mask_path = mask_dir + filename[0:-4]
            car_mask = cv.imread(mask_path + car_pattern)
            car_mask = cv.cvtColor(car_mask, cv.COLOR_BGR2GRAY)
            car_mask = cv.resize(car_mask, (256, 256))
            line_mask = cv.imread(mask_path + line_pattern)
            line_mask = cv.cvtColor(line_mask, cv.COLOR_BGR2GRAY)
            line_mask = cv.resize(line_mask, (256, 256))
            lane_mask = cv.imread(mask_path + lane_pattern)
            lane_mask = cv.cvtColor(lane_mask, cv.COLOR_BGR2GRAY)
            lane_mask = cv.resize(lane_mask, (256, 256))
            back_mask = cv.imread(mask_path + back_pattern)
            back_mask = cv.cvtColor(back_mask, cv.COLOR_BGR2GRAY)
            back_mask = cv.resize(back_mask, (256, 256))

            mask_list = []
            mask_list.append(car_mask)
            mask_list.append(line_mask)
            mask_list.append(lane_mask)
            mask_list.append(back_mask)
            #print(f'car_mask size: {car_mask.shape}')
            #print(car_mask)
            for chan in range(NUM_CHANNELS):
                for i in range(IM_HEIGHT):
                    for j in range(IM_WIDTH):
                        val = 1 if mask_list[chan][i, j] > 0 else 0
                        target_tensor[chan, i, j] = val
                        if val == 1:
                            for check in range(chan):
                                if mask_list[check][i, j] > 0:
                                    target_tensor[chan, i, j] = 0
        print(target_tensor.shape)
        print(target_tensor)
        
        rand = random.uniform(0, 1)

        if rand < 0.1:
            new_dir = resize_dir + '/test/' + filename
            this_dir = target_dir + '/test/' + filename[0:-4] + '_mask.pt'
            torch.save(target_tensor, this_dir)
            cv.imwrite(new_dir, resize_img)
        elif rand < 0.2:
            new_dir = resize_dir + '/val/' + filename
            this_dir = target_dir + '/val/' + filename[0:-4] + '_mask.pt'
            torch.save(target_tensor, this_dir)
            cv.imwrite(new_dir, resize_img)
        else:
            new_dir = resize_dir + '/train/' + filename
            this_dir = target_dir + '/train/' + filename[0:-4] + '_mask.pt'
            torch.save(target_tensor, this_dir)
            cv.imwrite(new_dir, resize_img)
        


