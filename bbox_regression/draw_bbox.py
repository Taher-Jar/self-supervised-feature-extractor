from PIL import Image, ImageDraw
import os
import math

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

image = Image.new('L', (100, 100), 127)
draw = ImageDraw.Draw(image)

square_center = (50,50)
square_length = 40

square_vertices = (
    (square_center[0] + square_length / 2, square_center[1] + square_length / 2),
    (square_center[0] + square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] + square_length / 2)
)

square_vertices = [rotated_about(x,y, square_center[0], square_center[1], math.radians(45)) for x,y in square_vertices]

draw.polygon(square_vertices, fill=255)

image.save("output.png")
#def draw_bbox(image_path, gt_box_coors, pred_box_coors):
#
#    image = Image.open(image_path).convert('RGB')
#    draw = ImageDraw.Draw(image)
#    draw.rectangle(gt_box_coors,outline="green", width=3)
#    image.save("bbox_image_test.png", format="png")
#
#
#draw_bbox("C:/Users/t.jarjanazi/Documents/self-supervised-dev/Data/MSRA-TD500/train/IMG_0183.JPG", (231.0,258.0,873.0,351.0), (231.0,258.0,873.0,351.0))