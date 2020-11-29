import os
import shutil

# script for splitting the data into train, test, and validation

if not os.path.exists("train"):
    os.makedirs("train")
    os.makedirs("train/yes")
    os.makedirs("train/no")

if not os.path.exists("test"):
    os.makedirs("test")
    os.makedirs("test/yes")
    os.makedirs("test/no")

if not os.path.exists("validation"):
    os.makedirs("validation")
    os.makedirs("validation/yes")
    os.makedirs("validation/no")

path = './preprocessed_dataset'
for f in os.listdir(path):
    images = os.listdir(f'{path}/{f}')
    for n, image in enumerate(images):
        if n < 0.05 * len(images):
            shutil.copy(f'{path}/{f}/{image}', f'test/{f}/{f + str(n)}.jpg')
        elif n < 0.8 * len(images):
            shutil.copy(f'{path}/{f}/{image}', f'train/{f}/{f + str(n)}.jpg')
        else:
            shutil.copy(f'{path}/{f}/{image}', f'validation/{f}/{f + str(n)}.jpg')
