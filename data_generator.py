import os
import sys
import random as random
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import textwrap
from string import ascii_letters
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

cwd = os.getcwd()

images_directory = os.path.sep.join([cwd, "Data/ImageNet/train.X1"])

images_paths = []
for subdir in os.listdir(images_directory):
    subdir_path = os.path.sep.join([images_directory, subdir])
    for image in os.listdir(subdir_path):
        image_path = os.path.sep.join([subdir_path, image])
        images_paths.append(image_path)

test_images_directory = os.path.sep.join([cwd, "Data/ImageNet/train.X2"])

test_images_paths = []
for subdir in os.listdir(test_images_directory):
    subdir_path = os.path.sep.join([test_images_directory, subdir])
    for image in os.listdir(subdir_path):
        image_path = os.path.sep.join([subdir_path, image])
        test_images_paths.append(image_path)

text_file_path = os.path.sep.join([cwd, "Data/English_phrases_and_sayings.csv"])

fonts_directory = os.path.sep.join([cwd, "Data/MS_fonts"])
    

random.seed(23)

###################################
##### Complex Data Generation #####
###################################

def make_complex_data(image_path, input_text):
    image = Image.open(image_path)

    w, h = image.size


    # font color, scale and thickness
    text_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)

    box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.choice([0,255]))
    font_size = random.randint(30, 50)
    font_scale = (font_size*10)/max(w,h)
    # define font face
    fonts = os.listdir(fonts_directory)
    font = os.path.sep.join([fonts_directory,random.choice(fonts)])
    font = ImageFont.truetype(font=font, size=int(font_size/font_scale))

    thickness = 1
    division_scalar1 = random.uniform(2, 4)
    division_scalar2 = random.uniform(2, 4)
    text = input_text

    # Calculate the average length of a single character of our font.
    # Note: this takes into account the specific font and font size.
    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    # Translate this average length into a character count
    max_char_count = int(w / avg_char_width)

    fill_text = textwrap.fill(text=text, width=max_char_count)

    wrap_text = textwrap.wrap(text=text, width=max_char_count)

    text_width_max = max([font.getsize(i)[0] for i in wrap_text])
    height_sum = 0

    for i in wrap_text:
        height_sum += (font.getsize(i)[1]*1.5)

    box_size = (text_width_max+20, int(height_sum))

    # create image with correct size
    box_img = Image.new('RGBA', box_size, box_color)

    box_draw = ImageDraw.Draw(box_img)
    box_draw.text((0, 0), fill_text, font=font, fill=text_color, stroke_width=thickness)

    angle = random.choice([45,90,135,180,225,270,315,360])
    
    box_img = box_img.rotate(angle, resample=Image.NEAREST, expand=True)

    image.paste(box_img,(int(w/division_scalar1),int(h/division_scalar2)), box_img)

    image  = image.resize((224,224))

    return image

phrases = open(text_file_path).read().strip().split("\n")


'''num_images = 16
bound_change = 0
for i in range(len(phrases)):
    for j in range((0+bound_change),(num_images+bound_change)):
        complex_image = make_complex_data(images_paths[j],phrases[i])
        complex_image.save(f"Data/FACT 2.0/train_data/complex_image_{i}_{j}.png", format="png")
        bound_change += 1'''
    
##################################
##### Simple Data Generation #####
##################################

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def make_simple_data(alphabet_letter):
    text_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
    box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
    fonts = os.listdir(fonts_directory)
    font = os.path.sep.join([fonts_directory,random.choice(fonts)])
    font = ImageFont.truetype(font=font, size=72)
    background_img = Image.new('RGBA', (224,224), box_color)

    w, h = background_img.size
    text = alphabet_letter
    #if random.random() <= 0.5:
    #    text = text.upper()

    random_scalar = random.uniform(1.2,4)

    background_draw = ImageDraw.Draw(background_img)
    background_draw.text(xy=(w / random_scalar, h / random_scalar), text=text, font=font, fill=text_color, anchor='mm')

    return background_img


'''for i in range(len(letters)):
    for j in range(500):
        simple_image = make_simple_data(letters[i])
        simple_image.save(f"Data/FACT 2.0/train_data/simple_image_{i}_{j}.png", format="png")'''

test_fonts_directory = os.path.sep.join([cwd, "Data/test_fonts"])

###################################
##### Test Data Generation #####
###################################

def make_test_data(image_path, input_text):
    image = Image.open(image_path)

    w, h = image.size


    # font color, scale and thickness
    text_color = (150, 250, 100, 255)

    box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 0)
    font_size = 50
    font_scale = (font_size*10)/max(w,h)
    # define font face
    fonts = os.listdir(test_fonts_directory)
    font = os.path.sep.join([test_fonts_directory,random.choice(fonts)])
    font = ImageFont.truetype(font=font, size=int(font_size/font_scale))

    thickness = 1
    division_scalar1 = 4
    division_scalar2 = 4
    text = input_text

    # Calculate the average length of a single character of our font.
    # Note: this takes into account the specific font and font size.
    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    # Translate this average length into a character count
    max_char_count = int(w / avg_char_width)

    fill_text = textwrap.fill(text=text, width=max_char_count)

    wrap_text = textwrap.wrap(text=text, width=max_char_count)

    text_width_max = max([font.getsize(i)[0] for i in wrap_text])
    height_sum = 0

    for i in wrap_text:
        height_sum += (font.getsize(i)[1]*1.5)

    box_size = (text_width_max+20, int(height_sum))

    # create image with correct size
    box_img = Image.new('RGBA', box_size, box_color)

    box_draw = ImageDraw.Draw(box_img)
    box_draw.text((0, 0), fill_text, font=font, fill=text_color, stroke_width=thickness)

    angle = 360
    
    box_img = box_img.rotate(angle, resample=Image.NEAREST, expand=True)

    image.paste(box_img,(int(w/division_scalar1),int(h/division_scalar2)), box_img)

    image  = image.resize((224,224))

    return image

'''num_test_images = 2
for i in range(2032,len(phrases)):
    for j in range(0,num_test_images):
        complex_image = make_test_data(test_images_paths[0],phrases[i])
        complex_image.save(f"Data/Test_data/synthetic data/test_image_{i}_{j}.png", format="png")'''


##################################
##### Digits Data Generation #####
##################################

digits = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for i in range(len(digits)):
    for j in range(1000):
        simple_image = make_simple_data(digits[i])
        simple_image.save(f"Data/FACT 2.0/train_data/digit_image_{i}_{j}.png", format="png")