#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch


N_IMAGES = 821
IMG_PATHS = ['az-45el0', 'az0el0', 'az45el0',
             'az90el0', 'az-90el0', 'az180el0', 'az0el-90', 'az0el-45', 'az0el45','az0el90']


def preprocess_images(IMG_PATH):

    if os.path.isfile(IMG_PATH+".pth"):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images ...")
    raw_images = []
    for i in range(2000):
        if i % 10000 == 0:
            print(i)
        img_path = '../AI_DATA/generated_face_10_train/{}/{}.jpg'.format(IMG_PATH, i)
        if os.path.isfile(img_path):
            raw_images.append(mpimg.imread(img_path))

    # if len(raw_images) <= N_IMAGES:
    #     raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))

    data = np.array(raw_images)
    data = torch.from_numpy(data)
    print(data.size())

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data, "../AI_DATA/generated_face_10_train/generated_"+IMG_PATH+".pth")



for p in IMG_PATHS:
    preprocess_images(p)
