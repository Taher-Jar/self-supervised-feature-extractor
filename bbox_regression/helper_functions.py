import numpy as np
import torch
from PIL import Image
import os




# load the contents of the CSV annotations file
def parse_data(images_path, annotations_path):
    rows = open(annotations_path).read().strip().split("\n")
    # initialize the list of data (images), our target output predictions
    # (bounding box coordinates), along with the filenames of the
    # individual images
    data = []
    targets = []
    filepaths = []
    coords_list = []

    for row in rows:
    	# break the row into the filename and bounding box coordinates
        row = row.split(",")
        (filename, startX, startY, endX, endY, angle) = row
        coords = startX, startY, endX, endY
        coords_list.append(coords)
        # derive the path to the input image, load the image and grab its dimensions
        imagePath = os.path.sep.join([images_path, filename])
        image = Image.open(imagePath).convert('RGB')
        old_w, old_h = image.size
        # resize the image and change the bbox coordinates accordingly
        image = image.resize((224, 224))
        w, h = image.size
        dimension_change_w = w / old_w
        dimension_change_h = h / old_h
        startX = float(startX) * dimension_change_w
        startY = float(startY) * dimension_change_h
        endX = float(endX) * dimension_change_w
        endY = float(endY) * dimension_change_h
    	# scale the bounding box coordinates relative to the spatial
    	# dimensions of the input image
        startX = startX / w
        startY = startY / h
        endX = endX / w
        endY = endY / h
        angle = float(angle)
        
        # load the image and preprocess it
        image = np.array(image)
    	# update our list of data, targets, and filenames
        data.append(image)
        targets.append((startX, startY, endX, endY, angle))
        filepaths.append(imagePath)

    return (data,targets,filepaths,coords_list)