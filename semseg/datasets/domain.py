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

import torch
import logging
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


class Domain(Dataset):
    CLASSES = ('Background', 'EX', 'HE', 'SE', 'MA')

    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])

    SMALL_OBJECT = [4]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False, **kwargs) -> None:
        super().__init__()
        # assert split in ['train', 'val', 'val']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload
        self.pairs = []

        imgs = glob(osp.join(root, self.split, 'image') + '/*.jpg')
        for img_path in imgs:
            lbl_path = img_path.replace('image', 'mask').replace('.jpg', '.png')
            data_pair = [
                io.read_image(img_path) if self.preload else img_path,
                io.read_image(lbl_path)[-1:] if self.preload else lbl_path,
            ]
            self.pairs.append(data_pair)

        assert len(self.pairs) > 0, f"No images found in {root}"
        logging.info(f"Found {len(self.pairs)} {split} images.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        indices = list(range(len(self.pairs)))
        random.shuffle((indices))
        frist_index, second_index = indices[0], indices[1]
        frist_image, frist_label = self.pairs[frist_index]
        if not self.preload:
            frist_image = io.read_image(frist_image)
            frist_label = io.read_image(frist_label)[-1:]

        second_image, second_label = self.pairs[second_index]
        if not self.preload:
            second_image = io.read_image(second_image)
            second_label = io.read_image(second_label)[-1:]

        sample = {'img': frist_image, 'other img': second_image, 'mask': frist_label, 'other mask': second_label}
        if self.transform:
            sample = self.transform(sample)

        frist_image = sample['img']
        second_image = sample['other img']
        frist_label = sample['mask']
        second_label = sample['other mask']

        return frist_image, second_image, torch.squeeze(frist_label.long()), torch.squeeze(second_label.long())


if __name__ == '__main__':
    from semseg.augmentations import RandomHorizontalFlip_multi, RandomVerticalFlip_multi, RandomResizedCrop_multi
    from torchvision import transforms

    transform = transforms.Compose([
        RandomResizedCrop_multi((512, 512), scale=(0.5, 2.0), seg_fill=0),
        RandomHorizontalFlip_multi(p=0.5),
        RandomVerticalFlip_multi(p=0.5)
    ])
    _dataset = Domain('/home_lv/jiaoli.liu/sosnet/dataset/domain12', 'train', transform=transform, preload=False)
    for fi, fl, si, sl in _dataset:
        print(fi.size(), fl.size(), si.size(), sl.size())
