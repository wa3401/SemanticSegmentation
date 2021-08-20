import torch
import torchvision
from model import UNET, DoubleConv
import torchvision.transforms as transforms
from imageData import ImageData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from loss import dice_loss
from collections import defaultdict
import time
import copy
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import helper



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 100
batch_size = 8
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor()]
)

train_img_dir = '/home/williamanderson/Semantic Segmentation/images/willResize/train/'
train_target_dir = '/home/williamanderson/Semantic Segmentation/images/willTargets/train/'
train_set = ImageData(train_img_dir, train_target_dir, transform)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_img_dir = '/home/williamanderson/Semantic Segmentation/images/willResize/val/'
val_target_dir = '/home/williamanderson/Semantic Segmentation/images/willTargets/val/'
val_set = ImageData(val_img_dir, val_target_dir, transform)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

test_img_dir = '/home/williamanderson/Semantic Segmentation/images/willResize/test/'
test_target_dir = '/home/williamanderson/Semantic Segmentation/images/willTargets/test/'
test_set = ImageData(test_img_dir, test_target_dir, transform)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

images, targets = next(iter(train_dataloader))
print(f"Train Feature batch shape: {images.size()}")
print(f"Train Labels batch shape: {targets.size()}")





dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

model = UNET(out_channels=4).to(DEVICE)

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

plt.imshow(reverse_transform(images[3]))
#plt.show()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, num_epochs):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs.to(DEVICE)
                labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(f'output size: {outputs.size()}')
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = train_model(model, optimizer, num_epochs)

torch.save(model, '/home/williamanderson/Semantic Segmentation/model/bestModel.pt')

model.eval()

# Get the first batch
for inputs, labels in test_dataloader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)




    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
