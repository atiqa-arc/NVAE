# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import numpy as np
from PIL import Image
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import urllib
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN
from medical_dataset import MedicalImageDataset


class StackedMNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        index1 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index2 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index3 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        self.num_images = 2 * len(self.data)

        self.index = []
        for i in range(self.num_images):
            self.index.append((index1[i], index2[i], index3[i]))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)

def download_omniglot(data_dir):
    filename = 'chardata.mat'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print('Downloaded', filename)

    return


def load_omniglot(data_dir):
    download_omniglot(data_dir)

    data_path = os.path.join(data_dir, 'chardata.mat')

    omni = loadmat(data_path)
    train_data = 255 * omni['data'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))
    test_data = 255 * omni['testdata'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))

    train_data = train_data.astype('uint8')
    test_data = test_data.astype('uint8')

    return train_data, test_data

def get_loaders_eval(dataset, args):
    """Get medical dataset loaders."""
    if dataset == 'medical':
        num_classes = 1
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        valid_transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        
        # Define CSV paths
        train_csv = os.path.join(args.data, "train_pairs.csv")
        val_csv = os.path.join(args.data, "val_pairs.csv")
        
        # Create datasets
        train_data = MedicalImageDataset(
            csv_path=train_csv,
            image_root=args.data,
            image_size=256,
            transform=train_transform,
        )
        valid_data = MedicalImageDataset(
            csv_path=val_csv,
            image_root=args.data,
            image_size=256,
            transform=valid_transform,
        )
        
        # Create DataLoaders
        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=True)
    
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)
    
        return train_queue, valid_queue, num_classes
    
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported")



def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


