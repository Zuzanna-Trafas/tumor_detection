import os
import shutil

# script for splitting the data into train, test, and validation

if not os.path.exists("train"):
    os.makedirs("train")

if not os.path.exists("test"):
    os.makedirs("test")

if not os.path.exists("validation"):
    os.makedirs("validation")

path = './brain_tumor_dataset'
for f in os.listdir(path):
    images = os.listdir(path + "/" + f)
    for i in range(len(images)):
        if i < 0.05 * len(images):
            shutil.copy(path + "/" + f + "/" + images[i], 'test/' + f + str(i) + ".jpg")
        elif i < 0.8 * len(images):
            shutil.copy(path + "/" + f + "/" + images[i], 'train/' + f + str(i) + ".jpg")
        else:
            shutil.copy(path + "/" + f + "/" + images[i], 'validation/' + f + str(i) + ".jpg")