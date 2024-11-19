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
import torch
import logging
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


class FGADR(Dataset):
    """UAVid2020 dataset.

    In segmentation map annotation for UAVid2020, 0 stands for background, which is
    included in 8 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png', too.
    In UAVid2020, 200 images for training, 70 images for validating, and 150 images for testing.
    The 8 classes and corresponding label color (R,G,B) are as follows:
        'label name'        'R,G,B'         'label id'
        Background clutter  (0,0,0)         0
        EX                  (128,0,0)       1
        HE                  (0,128,0)       2
        SE                  (128,128,0)     3
        MA                  (0,0,128)       4

    """

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

        imgs = glob(osp.join(root, 'image', self.split) + '/*.jpg')
        for img_path in imgs:
            lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
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
        image, label = self.pairs[index]
        if not self.preload:
            image = io.read_image(image)
            label = io.read_image(label)[-1:]

        # if self.split == 'train':
        #     image = self.RCT(image)

        if self.transform:
            image, label = self.transform(image, label)

        return image, torch.squeeze(label.long())

    # def RCT(self, image):
    #     a,b = 0.4, 1.2  # 0.4-1.6
    #     aug_scale1 = a + b * torch.rand(1).item()
    #     aug_scale2 = a + b * torch.rand(1).item()
    #     aug_scale3 = a + b * torch.rand(1).item()
    #     add_scale1 = -255 + 510 * torch.rand(1).item()
    #     add_scale2 = -255 + 510 * torch.rand(1).item()
    #     add_scale3 = -255 + 510 * torch.rand(1).item()
    #     image[:,:,0] = image[:,:,0]*aug_scale1 + add_scale1
    #     image[:,:,1] = image[:,:,1]*aug_scale2 + add_scale2
    #     image[:,:,2] = image[:,:,2]*aug_scale3 + add_scale3
    #     return image


if __name__ == '__main__':
    # _dataset = FGADR('../../data/FGADR', 'train', preload=False)
    # for _i, _l in _dataset:
    #     print(_i.size(), _l.size())

    from semseg.augmentations import RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, Compose, Normalize
    from torch.utils import data
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    transform = Compose([
        RandomResizedCrop((1024, 1024), scale=(0.5, 2.0), seg_fill=0),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    _dataset = FGADR('/home_lv/jiaoli.liu/sosnet/data/FGADR', 'train', transform=None, preload=False)
    palette = FGADR.PALETTE
    palette = palette / 255.0

    camp = ListedColormap(palette.numpy())

    bs = 6
    trainloader = data.DataLoader(_dataset, batch_size=bs, num_workers=0)
    for i, (image, label) in enumerate(trainloader):
        print(image.size(), label.size())

        # 可视化图像
        fig, axs = plt.subplots(bs, 2, figsize=(10, 5))

        for j in range(bs):
            # 显示原始图像
            img_show = image[j].permute(1, 2, 0).numpy()  # 转换为HWC格式

            if img_show.dtype == np.float32:  # 如果是浮点数
                img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())  # 归一化到 [0, 1]
            elif img_show.dtype == np.uint8:  # 如果是整数
                img_show = img_show  # 已经是 [0, 255] 范围，不需要转换
            else:
                img_show = img_show.astype(np.uint8)  # 转换为整数类型

            axs[j, 0].imshow(img_show)
            axs[j, 0].set_title(f'Image {j + 1}')
            axs[j, 0].axis('off')

            # 显示标签
            lbl_show = label[j].numpy()
            lbl_show = lbl_show.squeeze() if lbl_show.ndim > 2 else lbl_show
            # print("Label values:", np.unique(lbl_show))

            axs[j, 1].imshow(lbl_show, cmap=camp, vmin=0, vmax=len(FGADR.CLASSES) - 1)
            axs[j, 1].set_title(f'Label {j + 1}')
            axs[j, 1].axis('off')

        plt.show()
        if i == 0:  # 只显示第一批图像
            break

