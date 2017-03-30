import sys
sys.path.insert(0, '/usr/local/opt/opencv3/lib/python3.5/site-packages/')
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

src_dir = 'Letters_v2/A/'
output_dir_rot1 = 'rotation/'
output_dir_rot2 = 'Letters_v2/A/'


def rotate_and_change(src, rot):
    img = cv2.imread(src, 0)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=255)
    # threshold
    ret, thresh1 = cv2.threshold(dst, 120, 255, cv2.THRESH_BINARY)
    return thresh1


def rotate_one_image(src, output_dir):
    print(src)
    list_rotations = []
    list_rotations.extend(range(20))
    list_rotations.extend(range(340, 360))
    print(list_rotations)
    i = 1
    for j in list_rotations:
        im_out = rotate_and_change(src, j)

        cv2.imwrite(output_dir + str(i) + '.png', im_out)
        i += 1
    return None


def rotate_all_images(src_dir, output_dir, max_angle=15):
    files = os.listdir(src_dir)
    for file in files:
        i = 1
        list_rotations = []
        list_rotations.extend(range(max_angle))
        list_rotations.extend(range(360-max_angle, 360))
        for rot in list_rotations:
            im_out = rotate_and_change(src_dir + file, rot)
            cv2.imwrite(output_dir + file.split('.')[0] + '_' + str(i) + '.png', im_out)
            i += 1

    return None

    
def erode_dilate(src, output_dir, size=3):
    img = cv2.imread(src, 0)
    src = src.split('.')[0]
    i = 0
    for size in range(2,size+1):
        kernel = np.ones((size,size),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imwrite(output_dir + src + '_'+ str(i) + '.png', erosion)
        i += 1
        dilation = cv2.dilate(img,kernel,iterations = 1)
        cv2.imwrite(output_dir + src + '_'+ str(i) + '.png', dilation)
        i += 1


if __name__ == '__main__':
    # return 10 images with different rotation from one image (image fixed)
    #rotate_one_image(src_dir, output_dir_rot1)
    #erode_dilate(src_dir, output_dir_rot1)
    # return for all the images of a directory the images corresponding to a rotation (rotation fixed)
    rotate_all_images(src_dir, output_dir_rot2, 15)

