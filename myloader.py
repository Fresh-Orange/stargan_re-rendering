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
from TripletFaceDataset import TripletFaceDataset




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

    data_loader = Loader(transform, batch_size)
    return data_loader


def get_triplet_loader(config, image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    #if mode == 'train':
    #    transform.append(T.RandomHorizontalFlip())
    # transform.append(T.ToPILImage())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    # transform.append(T.RandomRotation((90,90)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_dir = TripletFaceDataset(dir=config.id_dataroot, n_triplets=config.n_triplets, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dir,
                                               batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_dir, train_loader


class Loader(object):

    def __init__(self, transform, batch_size):
        self.transform = transform
        self.images, self.catimages = self.load_images()
        self.batch_size = batch_size

    def load_images(self):
        """
        Load dataset.
        """
        # load data
        images_filename_1 = 'multipie_az-45el0.pth'
        images_filename_2 = 'multipie_az-90el0.pth'
        images_filename_3 = 'multipie_az0el0.pth'
        images_filename_4 = 'multipie_az180el0.pth'
        images_filename_5 = 'multipie_az45el0.pth'
        images_filename_6 = 'multipie_az90el0.pth'

        print("Reading data ...")

        print("Reading class 1")
        images1 = torch.load(os.path.join(DATA_PATH, images_filename_1))
        #print(images1.size())
        #sample_path = os.path.join('{}-images.jpg'.format(1))
        #utils.save_image(images1[0].permute(2, 0, 1), sample_path)
        #raise RuntimeError
        print("Reading class 2")
        images2 = torch.load(os.path.join(DATA_PATH, images_filename_2))
        print("Reading class 3")
        images3 = torch.load(os.path.join(DATA_PATH, images_filename_3))
        print("Reading class 4")
        images4 = torch.load(os.path.join(DATA_PATH, images_filename_4))
        print("Reading class 5")
        images5 = torch.load(os.path.join(DATA_PATH, images_filename_5))
        print("Reading class 6")
        images6 = torch.load(os.path.join(DATA_PATH, images_filename_6))
        assert len(images1) == len(images2) == len(images3)

        print("Finish reading data ...")

        images = [images1, images2, images3, images4, images5, images6]

        catimages = torch.cat((images1,images2, images3, images4, images5, images6), 0)

        return images, catimages


    def train_batch(self):
        idxs = np.random.randint(low=0, high=self.len(), size=self.batch_size)
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