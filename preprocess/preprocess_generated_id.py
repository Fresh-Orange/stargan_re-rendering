#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import re


N_IMAGES = 1000
IMG_SIZE = 256
ATTR_PATH = 'attributes.pth'


def preprocess_images():

    root_path = "/media/data2/laixc/AI_DATA/generated_face_10_train"
    classes = os.listdir(root_path)
    print("classes = ", classes)

    if not os.path.isdir("generated_id"):
        os.makedirs("generated_id")

    for class_ in classes:
        print("Reading images...")
        path = os.path.join(root_path, class_)
        for id in range(1010):
            img_path = os.path.join(path, "{}.jpg".format(id))
            if os.path.isfile(img_path):
                id_dir = os.path.join("generated_id", "{}".format(id))
                if not os.path.isdir(id_dir):
                    os.makedirs(id_dir)
                image = mpimg.imread(img_path)
                mpimg.imsave("generated_id/{}/{}.png".format(id, class_), image)
                print("saved in generated_id/{}/{}.png".format(id, class_))



preprocess_images()
#preprocess_attributes()
