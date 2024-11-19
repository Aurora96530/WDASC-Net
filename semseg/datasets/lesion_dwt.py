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
from pytorch_wavelets import DWTForward, DWTInverse
import math


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


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    #
    # # for totally repeat
    # style_std_dropout = style_std.detach()
    # style_mean_dropout = style_mean.detach()
    # return normalized_feat * style_std_dropout.expand(size) + style_mean_dropout.expand(size)
    return style_mean.expand(size) + (content_feat - content_mean.expand(size)) * style_std.expand(size) / content_std.expand(size)


def wavelet_DWT_and_styleswap_and_IWT(A_fea, B_fea, J=1):
    # wavelet: DWT
    xfm = DWTForward(J=J, wave='haar', mode='zero').cuda()
    A_fea_Yl, A_fea_Yh = xfm(A_fea)
    B_fea_Yl, B_fea_Yh = xfm(B_fea)

    # styleswap the A_yl and B_yl
    A_fea_Yl_styleswap = adain(A_fea_Yl, B_fea_Yl)
    B_fea_Yl_styleswap = adain(B_fea_Yl, A_fea_Yl)

    # wavelet: IWT1
    ifm = DWTInverse(wave='haar', mode='zero').cuda()
    A_fea_styleswap = ifm((A_fea_Yl_styleswap, A_fea_Yh))
    B_fea_styleswap = ifm((B_fea_Yl_styleswap, B_fea_Yh))
    # return A_fea_styleswap
    return A_fea_styleswap, B_fea_styleswap


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w))*L)).astype(int)     # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]    # top right
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]    # bottom left
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]  # bottom right

    # c_h = np.floor(h/2.0).astype(int)
    # c_w = np.floor(w/2.0).astype(int)
    # # print (b)
    # h1 = c_h-b
    # h2 = c_h+b+1
    # w1 = c_w-b
    # w2 = c_w+b+1
    #
    # ratio = random.randint(1, 10)/10
    #
    # amp_src[:, h1:h2, w1:w2] = amp_src[:, h1:h2, w1:w2] * ratio + amp_trg[:, h1:h2, w1:w2] * (1 - ratio)
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    # fft_src = torch.rfft(src_img.clone(), signal_ndim=2, onesided=False)
    # fft_trg = torch.rfft(trg_img.clone(), signal_ndim=2, onesided=False)
    fft_src = torch.fft.rfft2(src_img, dim=(-2, -1), norm="ortho")
    fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
    fft_trg = torch.fft.rfft2(trg_img, dim=(-2, -1), norm="ortho")
    fft_trg = torch.stack((fft_trg.real, fft_trg.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()
    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    # src_in_trg = torch.irfft(fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])

    real_part = fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    imaginary_part = fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()
    complex_tensor = real_part + 1j*imaginary_part
    src_in_trg = torch.fft.irfft2(complex_tensor, dim=(-2, -1), s=[imgH, imgW], norm="ortho")

    return src_in_trg



class Lesion_fourier(Dataset):

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
            # other_label = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            other_img = other_img.numpy()
            image = image.numpy()
            amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))  #tiqu other_img'amp
            img_freq = source_to_target_freq(image, amp_trg, L=0.1)
            img_freq = np.clip(img_freq, 0, 255).astype(dtype=np.uint8)
            image = torch.from_numpy(image).float()
            img_freq = torch.from_numpy(img_freq).float()
            return image, img_freq, torch.squeeze(label.long())


class Lesion_dwt(Dataset):

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
            other_idl = np.random.choice(other_id_path).replace('\n', '').split(' ')[1]
            other_img = io.read_image(os.path.join(self.base_dir, other_domain_name, other_id))
            other_label = io.read_image(os.path.join(self.base_dir, other_domain_name, other_idl))

            sample = {'img': image, 'other img': other_img, 'mask': label, 'other mask': other_label}
            if self.transform:
                sample = self.transform(sample)

            image = sample['img']
            other_img = sample['other img']
            label = sample['mask']
            other_label = sample['other mask']

            return image, other_img, torch.squeeze(label.long()), torch.squeeze(other_label.long())


if __name__ == '__main__':
    from semseg.augmentations import RandomHorizontalFlip_multi, RandomVerticalFlip_multi, RandomResizedCrop_multi
    from torchvision import transforms
    transform = transforms.Compose([
        RandomResizedCrop_multi((512, 512), scale=(0.5, 2.0), seg_fill=0),
        RandomHorizontalFlip_multi(p=0.5),
        RandomVerticalFlip_multi(p=0.5)
    ])
    _dataset = Lesion_dwt(base_dir='/home_lv/jiaoli.liu/sosnet/datasetn', transform=transform, split='train',
                      domain_idx_list=[1, 2], is_out_domain=False, test_domain_idx=0, preload=False)
    for img, oimg, label, olabel in _dataset:
        print(img.size(), oimg.size(), label.size(), olabel.size())
    # for _il, _l in _dataset:
    #     print(_il.size(), _l.size())

