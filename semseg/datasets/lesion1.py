"""
@Project : semantic-segmentation
@File    : uavid2020.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2022/4/30 下午8:02
@e-mail  : 1183862787@qq.com
"""
import os
import os.path as osp
import random

import numpy as np
import torch
import logging
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)

    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.randint(1, 10) / 10

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img  # .cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)


class Lesion_sigle(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx=None, **kwargs) -> None:
        super().__init__()
        # assert split in ['train', 'val', 'val']
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['domain1', 'domain2', 'domain3']
        self.domain_idx =domain_idx
        self.split = split

        # self.id_path = []

        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload

        if split == 'train':
            with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
                self.id_path = f.readlines()
        elif split == 'test':
            with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                self.id_path = f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self) -> int:
        return len(self.id_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        cur_domain_name = self.domain_name[self.domain_idx]
        id = self.id_path[index]

        if not self.preload:
            image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
            label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]
            if self.transform:
                image, label = self.transform(image, label)

            return image, torch.squeeze(label.long())


class Lesion_multi(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx_list=None, is_freq=True, is_out_domain=False, test_domain_idx=None, **kwargs) -> None:
        super().__init__()
        # assert split in ['train', 'val', 'val']
        self.domain_idx_list = domain_idx_list
        self.base_dir = base_dir
        self.split = split
        self.num = num
        self.transform = transform
        self.domain_name = ['domain1', 'domain2', 'domain3']
        self.is_freq = is_freq
        self.is_out_domain = is_out_domain
        self.test_domain_idx = test_domain_idx
        self.id_path = []

        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload

        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
        elif split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self) -> int:
        return len(self.id_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])
        id = self.id_path[index]
        cur_domain_name = id.split(' ')[0].split('/')[5]

        if not self.preload:
            image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
            label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]

            if self.transform:
                image, label = self.transform(image, label)

            if self.is_freq:
                domain_list = train_domain_name.copy()
                if self.is_out_domain:
                    domain_list.remove(cur_domain_name)
                other_domain_name = np.random.choice(domain_list, 1)[0]
                with open(os.path.join(self.base_dir, other_domain_name + '_train.list'), 'r') as f:
                    other_id_path = f.readlines()
                other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
                other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id)).resize((512, 512), Image.BILINEAR)

                other_img = other_img.numpy()
                image = image.numpy()
                amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))  #tiqu other_img'amp
                img_freq = source_to_target_freq(image, amp_trg, L=0.1)
                img_freq = np.clip(img_freq, 0, 255).astype(dtype=np.uint8)
                image = torch.from_numpy(image).float()
                img_freq = torch.from_numpy(img_freq).float()

                return image, img_freq, torch.squeeze(label.long())
            else:
                return image, torch.squeeze(label.long())


if __name__ == '__main__':
    from semseg.augmentations import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
    transform = Compose([
        RandomResizedCrop((512, 512), scale=(0.5, 2.0), seg_fill=0),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5)
    ])
    _dataset = Lesion_multi(base_dir='/home_lv/jiaoli.liu/SOSNet-master/dataset', transform=transform, split='train',
                      domain_idx_list=[1, 2], is_out_domain=True, test_domain_idx=0, preload=False, is_freq=False)
    # for _img, _il, _l in _dataset:
    #     print(_img.size(), _il.size(), _l.size())
    # _dataset = Lesion_sigle(base_dir='/home_lv/jiaoli.liu/sosnet/dataset', transform=transform, split='test',
    #                         domain_idx=0, preload=False)
    for _il, _l in _dataset:
        print(_il.size(), _l.size())

