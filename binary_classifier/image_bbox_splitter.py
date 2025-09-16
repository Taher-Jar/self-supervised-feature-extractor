from PIL import Image
import os

def imgcrop(output_path, input, annotation, xPieces, yPieces):
    filename, file_extension = os.path.splitext(input)
    X_start,Y_start,X_end,Y_end = annotation
    im = Image.open(input)
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    #var_X_start = X_start
    #var_Y_start = Y_start
    #var_X_end = 0
    #var_Y_end = 0
    bounding_box_parts = []
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            tile = im.crop(box)
            tile.save(output_path + 'test_image' + "-" + str(i) + "-" + str(j) + file_extension)
            tile_width, tile_height = tile.size

            tile_x1 = j * tile_width
            tile_y1 = i * tile_height
            tile_x2 = tile_x1 + tile_width
            tile_y2 = tile_y1 + tile_height

            # Check if the bounding box is completely contained within the tile
            if X_start >= tile_x1 and Y_start >= tile_y1 and X_end <= tile_x2 and Y_end <= tile_y2:
                bounding_box_parts.append((X_start,Y_start,X_end,Y_end))
            else:
                # Calculate the intersection of the bounding box with the current tile
                intersection_x1 = max(tile_x1, X_start)
                intersection_y1 = max(tile_y1, Y_start)
                intersection_x2 = min(tile_x2, X_end)
                intersection_y2 = min(tile_y2, Y_end)

                # Check if there is an intersection
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    bounding_box_parts.append((intersection_x1, intersection_y1, intersection_x2, intersection_y2))
                else:
                    bounding_box_parts.append(None)
            '''if (var_X_start < ((j + 1) * width)) and (((j + 1) * width) != imgwidth):
                var_X_end = (j + 1) * width
                var_X_start_temp = var_X_end
                bbox_present = True

            elif(var_X_start < ((j + 1) * width)) and (((j + 1) * width) == imgwidth):
                var_X_end = X_end
                bbox_present = True


            elif (var_Y_start < ((i + 1) * height)) and (((i + 1) * height) != imgheight):
                var_Y_end = (i + 1) * height
                var_Y_start_temp = var_Y_end
                bbox_present = True

            elif (Y_start < ((i + 1) * height)) and (((i + 1) * height) == imgheight):
                var_X_end = Y_end

            var_X_start = var_X_start_temp
            var_Y_start = var_Y_start_temp
            #print("bbox split", (var_X_start, var_Y_start, var_X_end, var_Y_end))
            print(bbox_present)'''
    print(bounding_box_parts)


cwd = os.getcwd()

train_images = os.path.sep.join([cwd, 'Data/MSRA-TD500/train'])

images = os.listdir(train_images)

test_image = os.path.sep.join([train_images, images[5]])

destination_path = os.path.sep.join([cwd,'Data\split_images/'])

imgcrop(destination_path,test_image, (231.0,258.0,873.0,351.0), 3, 3)


def bbox_split(annotation, xPieces, yPieces):
    pass