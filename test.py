import os
import torchvision.transforms as transforms
import torch
from PIL import Image
current_dir = os.getcwd()
#source_dir = current_dir + "/svt1/img/"

folder_path = os.path.sep.join([current_dir, 'Data/FACT 2.0/train_data'])

#folder = [os.path.sep.join([folder_path,x]) for x in os.listdir(folder_path)]
#
#transform = transforms.ToTensor()
#images = [Image.open(img) for img in folder]
#
#transformed_imgs = [transform(x) for x in images]
print(len(os.listdir(folder_path)))