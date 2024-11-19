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
import random
from math import sqrt


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



def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


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


class Lesion(Dataset):

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

        # if not self.preload:
        if self.split =='train':
            image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
            label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]
            sample = {'img': image, 'mask': label}

            domain_list = train_domain_name.copy()
            if self.is_out_domain:
                domain_list.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name + '_train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
            other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            sample = {'img': image, 'other img': other_img, 'mask': label}

            if self.transform:
                sample = self.transform(sample)

                image = sample['img']
                label = sample['mask']
                other_img = sample['other img']

                other_img = other_img.numpy()
                image = image.numpy()
                amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))  #tiqu other_img'amp
                img_freq = source_to_target_freq(image, amp_trg, L=0.1)
                img_freq = np.clip(img_freq, 0, 255).astype(dtype=np.uint8)
                image = torch.from_numpy(image).float()
                img_freq = torch.from_numpy(img_freq).float()

            return image, img_freq, torch.squeeze(label.long())


class Lesion_new(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx_list=None, is_freq=True, is_out_domain=False, test_domain_idx=None, alpha=1.0, **kwargs) -> None:
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
        self.alpha = alpha

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
            domain_list = train_domain_name.copy()
            if self.is_out_domain:
                domain_list.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name + '_train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
            other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            other_label = io.read_image(os.path.join(self.base_dir, other_domain_name,
                                                     other_id.replace('image', 'mask').replace('.jpg', '.png')))
            # sample = {'img': image, 'other img': other_img, 'mask': label, 'other mask': other_label}
            # if self.transform:
            #     sample = self.transform(sample)
            #
            # image = sample['img']
            # other_img = sample['other img']
            # label = sample['mask']
            # other_label = sample['other mask']

            img_s2o, img_o2s = colorful_spectrum_mix(image, other_img, alpha=self.alpha)

            label = torch.squeeze(label.long())
            other_label = torch.squeeze(other_label.long())
            image = image.float()
            other_img = other_img.float()
            img_s2o = torch.from_numpy(img_s2o).float()
            img_o2s = torch.from_numpy(img_o2s).float()

            img = [image, other_img, img_s2o, img_o2s]
            lbl = [label, other_label, label, other_label]
            return img, lbl


class Lesion_dwt1(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx_list=None, is_freq=True, is_out_domain=False, test_domain_idx=None, alpha=1.0, **kwargs) -> None:
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
        self.alpha = alpha

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

        # if self.split == 'train':
        if not self.preload:
            image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
            label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]
            domain_list = train_domain_name.copy()
            if self.is_out_domain:
                domain_list.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name + '_train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
            other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            other_label = io.read_image(os.path.join(self.base_dir, other_domain_name,
                                                     other_id.replace('image', 'mask').replace('.jpg', '.png')))
            sample = {'img': image, 'other img': other_img, 'mask': label, 'other mask': other_label}
            if self.transform:
                sample = self.transform(sample)

            image = sample['img']
            other_img = sample['other img']
            label = sample['mask']
            other_label = sample['other mask']

            image = image.float()
            other_img = other_img.float()
            label = torch.squeeze(label.long())
            other_label = torch.squeeze(other_label.long())

            return image, label, other_img, other_label

# class Lesion1(Dataset):
#
#     CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
#     PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
#     SMALL_OBJECT = [4]
#
#     def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
#                  domain_idx_list=None, test_domain_idx=None, **kwargs) -> None:
#         super().__init__()
#         # assert split in ['train', 'val', 'val']
#         self.domain_idx_list = domain_idx_list
#         self.base_dir = base_dir
#         self.split = split
#         self.num = num
#         self.transform = transform
#         self.domain_name = ['domain1', 'domain2', 'domain3']
#         self.test_domain_idx = test_domain_idx
#         self.id_path = []
#
#         self.n_classes = len(self.CLASSES)
#         self.ignore_label = 255
#         self.preload = preload
#
#         if split == 'train':
#             for domain_idx in self.domain_idx_list:
#                 with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
#                     self.id_path = self.id_path + f.readlines()
#         elif split == 'test':
#             for domain_idx in self.domain_idx_list:
#                 with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
#                     self.id_path = self.id_path + f.readlines()
#
#         self.id_path = [item.replace('\n', '') for item in self.id_path]
#
#         if self.num is not None:
#             self.id_path = self.id_path[:self.num]
#         print("total {} samples".format(len(self.id_path)))
#
#     def __len__(self) -> int:
#         return len(self.id_path)
#
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         train_domain_name = self.domain_name.copy()
#         train_domain_name.remove(self.domain_name[self.test_domain_idx])
#         id = self.id_path[index]
#         cur_domain_name = id.split(' ')[0].split('/')[5]
#
#         if not self.preload:
#             image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
#             label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]
#             print(image.dtype)
#
#             image = image.float()
#             print(image.dtype)
#             return image, torch.squeeze(label.long())


class Lesion1(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx_list=None, is_freq=True, is_out_domain=False, test_domain_idx=None, alpha=1.0, **kwargs) -> None:
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
        self.alpha = alpha

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

            if self.split == 'train':
                image = self.RCT(image)

            if self.transform:
                image, label = self.transform(image, label)

            label = torch.squeeze(label.long())
            # image = image.float()

            return image, label

    def RCT(self, image):
        a, b = 0.4, 1.2  # 0.4-1.6
        aug_scale1 = a + b * torch.rand(1).item()
        aug_scale2 = a + b * torch.rand(1).item()
        aug_scale3 = a + b * torch.rand(1).item()
        add_scale1 = -255 + 510 * torch.rand(1).item()
        add_scale2 = -255 + 510 * torch.rand(1).item()
        add_scale3 = -255 + 510 * torch.rand(1).item()
        image[:, :, 0] = image[:, :, 0]*aug_scale1 + add_scale1
        image[:, :, 1] = image[:, :, 1]*aug_scale2 + add_scale2
        image[:, :, 2] = image[:, :, 2]*aug_scale3 + add_scale3
        return image

class Lesion11(Dataset):

    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')
    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
    SMALL_OBJECT = [4]

    def __init__(self, base_dir: str, split: str = 'train', transform=None, preload=False, num=None,
                 domain_idx_list=None, test_domain_idx=None, **kwargs) -> None:
        super().__init__()
        # assert split in ['train', 'val', 'val']
        self.domain_idx_list = domain_idx_list
        self.base_dir = base_dir
        self.split = split
        self.num = num
        self.transform = transform
        self.domain_name = ['domain1', 'domain2', 'domain3']
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
        print(cur_domain_name)

        if not self.preload:
            image = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
            label = io.read_image(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1]))[-1:]

            if self.transform:
                image, label = self.transform(image, label)
            return image, torch.squeeze(label.long())


class Lesion2(Dataset):

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

            domain_list = train_domain_name.copy()
            if self.is_out_domain:
                domain_list.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name + '_train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
            # other_id1 = other_id.replace('.jpg', '.png')
            other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            # other_label = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))[-1:]
            other_img = other_img.numpy()
            image = image.numpy()
            amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))  #tiqu other_img'amp
            img_freq = source_to_target_freq(image, amp_trg, L=0.1)
            img_freq = np.clip(img_freq, 0, 255).astype(dtype=np.uint8)
            image = torch.from_numpy(image).float()
            img_freq = torch.from_numpy(img_freq).float()
            # if self.transform:
            #     image, img_freq, label = self.transform(image, img_freq, label)
            return img_freq, torch.squeeze(label.long())


if __name__ == '__main__':
    from semseg.augmentations import RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, Compose
    from semseg.augmentations import RandomHorizontalFlip_multi, RandomVerticalFlip_multi, RandomResizedCrop_multi
    from torchvision import transforms
    # transform = transforms.Compose([
    #     RandomResizedCrop_multi((1024, 1024), scale=(0.5, 2.0), seg_fill=0),
    #     RandomHorizontalFlip_multi(p=0.5),
    #     RandomVerticalFlip_multi(p=0.5)
    # ])
    transform = Compose([
        RandomResizedCrop((1024, 1024), scale=(0.5, 2.0), seg_fill=0),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5)
    ])
    _dataset = Lesion1(base_dir='/home_lv/jiaoli.liu/sosnet/dataset', transform=transform, split='train',
                      domain_idx_list=[1, 2], is_out_domain=True, test_domain_idx=0, preload=False)

    print(len(_dataset))
    # for _img, _il, oi, ol in _dataset:
    #     print(_img.size(), _il.size(), oi.size(), ol.size())
    # _dataset = Lesion_sigle(base_dir='/home_lv/jiaoli.liu/sosnet/dataset', transform=transform, split='test',
    #                         domain_idx=0, preload=False)
    for _il, _l in _dataset:
        print(_il.size(), _l.size())

