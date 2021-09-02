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
#### dataTest.py
- A test file that was used before we switched labeling softwares becuase LabelBox API did not integrate with our system well
#### helper.py
- This file contains helper functions that are used to plot and show images such as the array of images seen above
#### imageData.py
- Defines a dataset class used for training the UNET model
#### loss.py
- Defines a function dice_loss that helps to computes the pixelwise cross entropy loss
#### main.py
- The main file that is run to complete training
- Consits of dataset creation, training loop with validation, model saving, and test loop
#### semanticSplit.py
- Script used to split image files between Jamie and myself for labeling purposes
