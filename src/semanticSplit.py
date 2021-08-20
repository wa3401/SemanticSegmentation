import os
import cv2 as cv
import fnmatch
import random


data_dir = '/home/williamanderson/Semantic Segmentation/updatedImages/'
file_list = os.listdir(data_dir)
# print(file_list)
pattern = "*.jpg"

for filename in file_list:
    print(filename)
    if fnmatch.fnmatch(filename, pattern):
        origPath = data_dir + filename
        img = cv.imread(origPath)
        rand = random.uniform(0,1)
        if  rand >= 0.3:
            path = '/home/williamanderson/Semantic Segmentation/willImages/' + filename
            cv.imwrite(path, img)
        else:
            path = '/home/williamanderson/Semantic Segmentation/jamieImages/' + filename
            cv.imwrite(path, img)