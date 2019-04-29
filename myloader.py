# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
import random
from torchvision import transforms as T
from torchvision import utils




DATA_PATH = "../AI_DATA/multipie_align_train"







def normalize_images(images):
    """
    Normalize image values.
    """
    return images.float().div_(255.0).mul_(2.0).add_(-1)


class DataSampler(object):

    def __init__(self, images, attributes,  params):
        """
        Initialize the data sampler with training data.
        """
        self.images = images
        self.batch_size = params.batch_size
        self.v_flip = params.v_flip
        self.h_flip = params.h_flip
        self.attributes = attributes

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.images.size(0)

    def train_batch(self, bs):
        """
        Get a batch of random images with their attributes.
        """
        # image IDs
        idx = torch.LongTensor(bs).random_(len(self.images))

        # select images / attributes
        batch_x = normalize_images(self.images.index_select(0, idx).cuda())
        batch_y = self.attributes.index_select(0, idx).cuda()

        # data augmentation
        if self.v_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(2, torch.arange(batch_x.size(2) - 1, -1, -1).long().cuda())
        if self.h_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(3, torch.arange(batch_x.size(3) - 1, -1, -1).long().cuda())

        return Variable(batch_x, volatile=False), Variable(batch_y, volatile=False)

    def eval_batch(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images[i:j].cuda())
        batch_y = self.attributes[i:j].cuda()
        return Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    #if mode == 'train':
    #    transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToPILImage())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    # transform.append(T.RandomRotation((90,90)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    global DATA_PATH
    DATA_PATH = image_dir

    data_loader = Loader(transform, batch_size)

    return data_loader


class Loader(object):

    def __init__(self, transform, batch_size):
        self.transform = transform
        self.images, self.catimages = self.load_images()
        self.batch_size = batch_size
        print("len", self.len())

    def load_images(self):
        """
        Load dataset.
        """
        images_filename = ['multipie_az-45el0.pth', 'multipie_az-90el0.pth', 'multipie_az0el0.pth',
                           'multipie_az180el0.pth', 'multipie_az45el0.pth', 'multipie_az90el0.pth']

        # images_filename = ['generated_az-45el0.pth', 'generated_az-90el0.pth', 'generated_az0el0.pth',
        #                    'generated_az180el0.pth', 'generated_az45el0.pth', 'generated_az90el0.pth',
        #                    'generated_az0el-90.pth',
        #                    'generated_az0el-45.pth', 'generated_az0el45.pth', 'generated_az0el90.pth'
        #                    ]

        # load data
        print("Reading data ...")

        images = []

        for filename in images_filename:
            print("Reading class {}".format(filename))
            images.append(torch.load(os.path.join(DATA_PATH, filename)))

        print("Finish reading data ...")

        catimages = torch.cat(tuple(images), 0)

        return images, catimages


    def train_batch(self):
        idxs = np.random.randint(low=0, high=self.len(), size=self.batch_size)
        # idxs = np.arange(0, self.batch_size)
        class_idxs = idxs // len(self.images[0])
        file_idxs = idxs % len(self.images[0])

        idxs = torch.LongTensor(idxs)
        images = self.catimages.index_select(0, idxs)

        images = [image.permute(2, 0, 1) for image in images]
        images = torch.stack([self.transform(image) for image in images])
        #images = self.transform(images)

        return images, torch.tensor(class_idxs), torch.tensor(file_idxs)

    def __getitem__(self, index):
        index = np.ones(shape=(1))*index
        #print(index)
        idxs = torch.LongTensor(index)
        class_idxs = idxs // len(self.images[0])

        images = self.catimages.index_select(0, idxs)

        images = [image.permute(2, 0, 1) for image in images]
        images = torch.stack([self.transform(image) for image in images])

        return images, torch.tensor(class_idxs)

    def test_batch(self):
        """
        batch_size = 1
        """
        idxs = np.random.randint(low=0, high=self.len(), size=1)
        class_idxs = idxs // len(self.images[0])
        file_idxs = idxs % len(self.images[0])

        idxs = torch.LongTensor(idxs)
        images = self.catimages.index_select(0, idxs)

        images = [image.permute(2, 0, 1) for image in images]
        images = torch.stack([self.transform(image) for image in images])
        #images = self.transform(images)

        return images, torch.tensor(class_idxs), torch.tensor(file_idxs)

    def get_by_idx(self, class_idxs, file_idxs):
        idxs = [class_idxs[i] * len(self.images[0]) + file_idxs[i] for i in range(len(class_idxs))]

        idxs = torch.LongTensor(idxs)
        images = self.catimages.index_select(0, idxs)

        images = [image.permute(2, 0, 1) for image in images]
        images = torch.stack([self.transform(image) for image in images])
        #images = self.transform(images)

        return images

    def len(self):
        return sum([len(ims) for ims in self.images])