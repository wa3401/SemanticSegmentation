# Semantic Segmentation
## Built by Will Anderson for use in the NIMBUS Lab F1Tenth Autonomus Driving Project
### Contact me at wanderson13@huskers.unl.edu with any questions or comments

## Description
- This package contains a full framework for training a Convolutional Neural Network (UNET Model) for use in an F1Tenth Autonomus Car
- The image datasets (held in images/willResize and images/willTargets) were gathered from the front facing Intel D435i Depth camera on the car
- These images were labeled by hand using an aplication for MacOS called RectLabel

  ### The 4 Labels 
  1. Current Lane
  2. Lane Line
  3. F1Tenth Car
  4. Background
  
## Test Set Results After Training
From Left to Right - Input, Ground Truth, Model Output
![Training Results](/testResults.png)
- Results could be improved with the addition of more training images, however this is very time consuming
- Once tested on the car, reconsider labeling more images

## /src Folder
#### combineMasks.py
- Takes each of the 4 layers of every mask image then saves them as 256x256x4 tensors for use as the ground truth
#### 
