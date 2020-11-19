import os

import cv2
import imutils
import numpy as np

GAUSSIAN_BLUR_SIZE = (5, 5)
THRESHOLD = 20
ERODE_ITERATIONS = 2
DILATE_ITERATIONS = 2
KERNEL = np.ones((3, 3), np.uint8)
EXTRA_SPACE = 0
RESULT_PATH = "./preprocessed_dataset"
SOURCE_PATH = "./brain_tumor_dataset"


def crop_image(image_name, result_name):
    img = cv2.imread(image_name)
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(greyscale_img, GAUSSIAN_BLUR_SIZE, 0)
    threshold_img = cv2.threshold(blurred_img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    eroded_img = cv2.erode(threshold_img, KERNEL, iterations=ERODE_ITERATIONS)
    dilated_img = cv2.dilate(eroded_img, KERNEL, iterations=DILATE_ITERATIONS)
    contours = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_contour = max(contours, key=cv2.contourArea)
    left_point = max(min(max_contour[:, :, 0])[0] - EXTRA_SPACE, 0)
    right_point = min(max(max_contour[:, :, 0])[0] + EXTRA_SPACE, len(img[0]))
    top_point = max(min(max_contour[:, :, 1])[0] - EXTRA_SPACE, 0)
    bottom_point = min(max(max_contour[:, :, 1])[0] + EXTRA_SPACE, len(img))
    result_img = img[top_point:bottom_point, left_point:right_point]
    cv2.imwrite(result_name, result_img)


if __name__ == '__main__':

    if not os.path.exists(f'{RESULT_PATH}'):
        os.makedirs(f'{RESULT_PATH}')

    if not os.path.exists(f'{RESULT_PATH}/yes'):
        os.makedirs(f'{RESULT_PATH}/yes')

    if not os.path.exists(f'{RESULT_PATH}/no'):
        os.makedirs(f'{RESULT_PATH}/no')

    for directory in os.listdir(SOURCE_PATH):
        images = os.listdir(f'{SOURCE_PATH}/{directory}')
        for n, image_name in enumerate(images):
            crop_image(f'{SOURCE_PATH}/{directory}/{image_name}',
                       f'{RESULT_PATH}/{directory}/{str(n)}.jpg')
