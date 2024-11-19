import torch
from torch import Tensor, nn
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead, UPerHead, SegFormerHead_fa


class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

        # self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)

class FSL(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32):
        """
        Fine-grained semantic learning module
        Parameters
        ----------
        seg_ch (int): numcer of channels for segmentation features
        sr_ch (int): number of channels for super-resolution
        """
        super(FSL, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, hidden_state, 1, 1),
            nn.ReLU()
        )
        self.conv_sp1 = nn.Conv2d(hidden_state, hidden_state,
                                 (7, 1), padding=(3, 0), bias=False)
        self.conv_sp2 = nn.Conv2d(hidden_state, hidden_state,
                                 (1, 7), padding=(0, 3), bias=False)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_state, seg_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        concat = torch.cat([sr_fe, seg_fe], dim=1)
        conv = self.conv1(concat)
        sp1 = self.conv_sp1(conv)
        sp2 = self.conv_sp2(conv)
        seg_fusion = self.fusion(sp1+sp2)
        return seg_fusion


class SegFormer0(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


class SegFormer_fa(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.head_bootom = SegFormerHead(self.backbone.channels,
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                channel=32,
                                num_classes=2,
                                scales=(1, 2, 3, 6))

        self.SRdecoder = SegFormerHead_fa(self.backbone.channels,
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)


        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 3, 1),
            torch.nn.BatchNorm2d(3),  # 添加了BN层
            torch.nn.ReLU(inplace=True)
        )

        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_edsr_1 = EDSRConv(64, 64)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_edsr_2 = EDSRConv(32, 32)
        self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up_edsr_3 = EDSRConv(16, 16)
        self.up_conv_last = nn.Conv2d(16, 3, 1)

        self.query = FSL(3, num_classes)
    def forward(self, x: Tensor):
        f_x4, f_x8, f_x16, f_x32 = self.backbone(x)
        logits_bottom = self.head_bootom([f_x4, f_x8, f_x16, f_x32])   # 4x reduction in image size
        logits_bottom = F.interpolate(logits_bottom, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            logits_bottom = F.interpolate(logits_bottom, scale_factor=2, mode='bilinear', align_corners=True)

            logits_sr = self.SRdecoder([f_x4, f_x8, f_x16, f_x32])

            logits_sr_up = self.up_sr_1(logits_sr)
            logits_sr_up = self.up_edsr_1(logits_sr_up)
            logits_sr_up = self.up_sr_2(logits_sr_up)
            logits_sr_up = self.up_edsr_2(logits_sr_up)
            logits_sr_up = self.up_sr_3(logits_sr_up)
            logits_sr_up = self.up_edsr_3(logits_sr_up)
            logits_sr_up = self.up_conv_last(logits_sr_up)

            seg_weight = self.query(logits_sr_up, logits_bottom)
            fusion_seg = seg_weight * logits_bottom + logits_bottom
            # logits_edge = self.head_edge(f_x4, f_x8)
            # logits_edge = F.interpolate(logits_edge, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_top = self.head_top([f_x4, f_x8, f_x16, f_x32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_top = F.interpolate(logits_top, scale_factor=2, mode='bilinear', align_corners=True)
            # return torch.cat([logits_seg, logits_so], dim=1), logits_edge
            return logits_bottom, logits_top, logits_sr_up, fusion_seg, None
            # return logits_bottom, logits_top, None

        return logits_bottom.contiguous()


if __name__ == '__main__':
    model = SegFormer_fa('MiT-B0', num_classes=8)
    model.train(True)
    model.init_pretrained('../../checkpoints/backbones/mit/mit_b0.pth')
    x = torch.zeros(4, 3, 512, 1024)
    y = model(x)
    if model.training:
        print(y[0].shape, y[1].shape)
    else:
        print(y.shape)