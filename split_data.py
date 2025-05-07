import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# Get all folder names in the data folder (that is, the class name that needs to be classified)
file_path = './FGVC2021'
leaf_class = [cla for cla in os.listdir(file_path)]

# Create a training set train folder
mkfile('split-data/fgvc8_train')
for cla in leaf_class:
    mkfile('split-data/fgvc8_train/' + cla)

# Create a testing set train folder
mkfile('split-data/fgvc8_test')
for cla in leaf_class:
    mkfile('split-data/fgvc8_test/' + cla)

# Dividing the dataset, training set: testing set = 8: 2
split_rate = 0.2

# Iterate through all images of all categories and divide them into training and validation sets in proportion
for cla in leaf_class:
    cla_path = file_path + '/' + cla + '/' 
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate)) 
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'split-data/fgvc8_test/' + cla
            copy(image_path, new_path)

        else:
            image_path = cla_path + image
            new_path = 'split-data/fgvc8_train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")
