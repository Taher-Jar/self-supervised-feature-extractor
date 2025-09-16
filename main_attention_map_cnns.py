import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import torch
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import os
import sys
import timm
from src.backbones.mobilenet import Backbone



cwd = os.getcwd()
#model = torch.load((cwd + '/model_dir/backbone/model9.pth'))
model = Backbone("mobilenetv2_100", pretrain=True)
model.eval()

#target_layers = [model.blocks[-1]]
#
## Read a PIL image
img = np.array(Image.open(cwd + '/Data/MSRA-TD500/test/IMG_0080.JPG').convert('RGB'))

img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
targets = None
target_layers = [model.backbone.conv_head]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
test_image = Image.fromarray(images)
test_image.save('test_pic3_fact2.0_new.png')