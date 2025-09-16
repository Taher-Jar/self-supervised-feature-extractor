import numpy as np
import torch
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import MLFlowLogger
import sys
from torchvision.ops import complete_box_iou_loss
from helper_functions import parse_data
from Bbox_model import Bbox
import timm

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from src.backbones.mobilenet import Backbone


cwd = os.getcwd()

bbox_dir = os.path.sep.join([cwd, 'bbox_regression'])

train_annotations = os.path.sep.join([cwd, 'Data/MSRA-annotations-train.csv'])
train_images = os.path.sep.join([cwd, 'Data/MSRA-TD500/train'])


# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-3
NUM_EPOCHS = 25
BATCH_SIZE = 1


# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]

train_data = np.array(parse_data(train_images,train_annotations)[0], dtype="float32") / 255.0
train_targets = np.array(parse_data(train_images,train_annotations)[1], dtype="float32")

tensor_x = torch.Tensor(train_data) 
tensor_y = torch.Tensor(train_targets)


my_dataset = TensorDataset(tensor_x,tensor_y) 
train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE)


model = torch.load(os.path.sep.join([cwd, '/model_dir/backbone/model9.pth']))

#model = Backbone("mobilenetv2_100", pretrain=True)

head = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=1280, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128,out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=5),
    nn.Sigmoid()
)


bbox_model = Bbox(model,head)

bbox_model.backbone.requires_grad_(False)
bbox_model.head.requires_grad_(True)

optimizer = torch.optim.Adam([param for param in bbox_model.head.parameters()], lr=INIT_LR)

train_losses = []
train_counter = []


criterion = nn.SmoothL1Loss()
def train(epoch):
    bbox_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.permute(0,3,2,1)
        optimizer.zero_grad()
        output = bbox_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.mean().item()))
            train_losses.append(loss.mean().item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            

for epoch in range(NUM_EPOCHS):
    train(epoch)

torch.save(bbox_model, os.path.join(bbox_dir, "output/model.pth"))