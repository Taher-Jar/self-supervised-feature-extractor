import matplotlib as plt
import numpy as np
import torch
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import generalized_box_iou_loss, generalized_box_iou, complete_box_iou_loss, complete_box_iou, box_iou
from helper_functions import parse_data
from Bbox_model import Bbox
from PIL import Image, ImageDraw
import math

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)


cwd = os.getcwd()

bbox_dir = os.path.sep.join([cwd, 'bbox_regression'])

test_annotations = os.path.sep.join([cwd, 'Data/MSRA-annotations-test.csv'])
test_images = os.path.sep.join([cwd, 'Data/MSRA-TD500/test'])


test_data = np.array(parse_data(test_images,test_annotations)[0], dtype="float32") / 255.0
test_targets = np.array(parse_data(test_images,test_annotations)[1], dtype="float32")

tensor_x = torch.Tensor(test_data) 
tensor_y = torch.Tensor(test_targets)

my_dataset = TensorDataset(tensor_x,tensor_y) 
test_loader = DataLoader(my_dataset)


network = torch.load(os.path.sep.join([bbox_dir, '/output/model.pth']))

test_losses = []
iou_scores = []

outputs = []

def test():
    loss_metric = complete_box_iou_loss
    performance_metric = complete_box_iou
    network.eval()
    test_loss_sum = 0
    iou_score_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.permute(0,3,2,1)
            output = network(data)
            outputs.append(output)
            test_loss = loss_metric(output[:,torch.LongTensor([0,1,2,3])], target[:,torch.LongTensor([0,1,2,3])]).abs()
            test_loss_sum += test_loss
            iou_score = performance_metric(output[:,torch.LongTensor([0,1,2,3])],target[:,torch.LongTensor([0,1,2,3])]).abs()
            iou_score_sum += iou_score


            test_losses.append(test_loss)
            iou_scores.append(iou_score)

    mean_loss = test_loss_sum / len(test_loader.dataset)
    mean_iou = iou_score_sum / len(test_loader.dataset)
    print(f"Average loss: {mean_loss} \n Average Intersection over Union: {mean_iou}")


test()
#
#print('predictions: ', outputs)
#print("losses: ",test_losses)
print("IOU: ",iou_scores)

#finds the straight-line distance between two points
def distance(ax, ay, bx, by):
    return math.sqrt((by - ay)**2 + (bx - ax)**2)

#rotates point `A` about point `B` by `angle` radians clockwise.
def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax,ay,bx,by)
    angle += math.atan2(ay-by, ax-bx)
    return (
        round(bx + radius * math.cos(angle)),
        round(by + radius * math.sin(angle))
    )


def draw_bbox(image_path, gt_box_coors, pred_box_coors):

    image = Image.open(image_path).convert('RGB')
    image = image.resize((224,224))
    gt_box_coors = [float(x)*224 for x in gt_box_coors]
    pred_box_coors = [float(x)*224 for x in pred_box_coors]
    print(gt_box_coors)
    print(pred_box_coors)
    draw = ImageDraw.Draw(image)
    draw.rectangle(gt_box_coors,outline="green", width=1)
    draw.rectangle(pred_box_coors,outline="red", width=1)
    image.save("bbox_image7.png", format="png")

i = 1
image_path = parse_data(test_images,test_annotations)[2][i]
gt_coors = list(parse_data(test_images,test_annotations)[1][i])[:4]
pred_coors = outputs[i][:,torch.LongTensor([0,1,2,3])][0]
draw_bbox(image_path, gt_coors, pred_coors)