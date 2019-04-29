#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import re


N_IMAGES = 1000
IMG_SIZE = 256
ATTR_PATH = 'attributes.pth'


def preprocess_images():

    num2class = {"00":"multipie_az180el0",
                 "01":"multipie_az90el0",
                 "04": "multipie_az45el0",
                 "07": "multipie_az0el0",
                 "10": "multipie_az-45el0",
                 "13": "multipie_az-90el0"}

    if not os.path.isdir("multi_pie_id"):
        os.makedirs("multi_pie_id")

    for num, class_ in num2class.items():
        print("Reading images...")
        sessions = ['01', '02', '03', '04']
        raw_images = []
        root_path = '/public/share/dataset/CMU-Multi-PIE/all/'
        for se in sessions:
            se_path = root_path + 'session' + se + '/multiview'
            dirs = os.listdir(se_path)
            for dir in dirs:
                if re.match('[0-9]{3}', dir):
                    if not os.path.isdir("multi_pie_id/{}".format(dir)):
                        os.makedirs("multi_pie_id/{}".format(dir))
                    path = os.path.join(se_path, dir)
                    path = os.path.join(path, '01')
                    path = os.path.join(path, '05_1')
                    position = num  # ################### 07光照在正前方 00无光照 01右边 04右中 10左中 13左边
                    path = os.path.join(path, '_'.join([dir, se, '01', '051', position]) + '.png')
                    image = mpimg.imread(path)
                    mpimg.imsave("multi_pie_id/{}/{}_{}.png".format(dir, se, position), image)
                    print("saved in multi_pie_id/{}/{}_{}.png".format(dir, se, position))


preprocess_images()
#preprocess_attributes()
