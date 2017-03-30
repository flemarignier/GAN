import cv2
import numpy as np
from os import listdir
import random


sample_dir = 'EnglishHnd/English/Hnd/Img/Sample020/'
out_dir = 'Letters/J/'
out_size = (28, 28)

def loadImage(src, col, show=False):
    img = cv2.imread(src,col)
    width, height = img.shape[:2]
    if show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img, width, height
    
  
def resize(img, shape, out_dir, show=False):
    im_out = cv2.resize(img, shape)
    if show:
        cv2.imshow('resized', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite(out_dir, im_out)
    return im_out
    
    
def find_limits(img, width, height):
    zeros = np.where(img == 0)
    l1 = min(zeros[0])
    l2 = max(zeros[0])
    c1 = min(zeros[1])
    c2 = max(zeros[1])
    return l1, l2, c1, c2


def cut_img(img, width, height):
    l1, l2, c1, c2 = find_limits(img, width, height)
    l = l2 - l1
    c = c2 - c1
    if c > l :
        delta = int((c - l)/2) + 10
        img = img[l1 - delta : l2 + delta, c1 - 10 : c2 + 10]
    else:
        delta = int((l - c)/2) + 10
        img = img[ l1 - 10 : l2 + 10, c1 - delta : c2 + delta]
    return img
    
    
def resize_all_images(sample_dir, out_dir, out_size):
    files = listdir(sample_dir)
    i = 1
    for file in files:
        img, width, height = loadImage(sample_dir + file , 0)
        img = cut_img(img, width, height)
        out = resize(img, out_size, out_dir + str(i) + '.png')
        i += 1
    
    
if __name__ == '__main__':
    #process_images(sample_dir, out_dir, out_size)
    img, width, height = loadImage('out.png', 0)
