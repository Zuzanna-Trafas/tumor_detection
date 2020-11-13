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
    images = os.listdir(f'{path}/{f}')
    for n, image in enumerate(images):
        if n < 0.05 * len(images):
            shutil.copy(f'{path}/{f}/{image}', f'test/{f + str(n)}.jpg')
        elif n < 0.8 * len(images):
            shutil.copy(f'{path}/{f}/{image}', f'train/{f + str(n)}.jpg')
        else:
            shutil.copy(f'{path}/{f}/{image}', f'validation/{f + str(n)}.jpg')
