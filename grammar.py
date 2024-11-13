import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y,_ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class change(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y,_ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,change=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.change=change

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split=="train" or self.split=="val":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir+"/"+slice_name+'.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2,0,1)
            label = torch.from_numpy(label.astype(np.float32))
        '''CutMix'''
        prob = 20  # 将 prob 设置为 0 即可关闭 CutMix
        if random.randint(0, 99) < prob:
            rand_slice_name = random.sample(self.sample_list, 1).strip('\n')
            rand_data_path = self.data_dir + "/" + rand_slice_name + '.npz'
            rand_data = np.load(rand_data_path)
            rand_image, rand_label = rand_data['image'], rand_data['label']
            H, W = rand_label.size()
            hori_vert = random.randint(0,3)
            if hori_vert == 0:
                half_image = image[:H/2, :, :]
                half_rand_image = rand_image[:H/2, :, :]
                image = torch.cat((half_image, half_rand_image), 0)
                half_label = label[:H/2, :]
                half_rand_label = rand_label[:H/2, :]
                label = torch.cat((half_label, half_rand_label), 0)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.change:
            sample = self.change(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
