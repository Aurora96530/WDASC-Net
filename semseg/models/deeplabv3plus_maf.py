"""
@Project : semantic-segmentation
@File    : deeplabv3plus_fa.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2022/5/17 下午8:39
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision

from semseg.models.base import BaseModel
from semseg.models.heads import UPerHead
from torch.nn import BatchNorm2d


class SSC(nn.Module):
    def __init__(self, in_ch, groups=4):
        """

        Args:
            in_ch (int): number of channels for input
            groups (int, Optional): Number of groups, Defatults to 4.
        """
        super(SSC, self).__init__()
        assert in_ch % groups == 0
        group_ch = in_ch // groups
        self.group_ch = group_ch
        self.conv = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1, 0)
        ])
        for i in range(1, groups):
            self.conv.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, bias=False)
            )
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        groups = torch.split(x, self.group_ch, dim=1)
        features = []
        for i, group in enumerate(groups):
            features.append(self.conv[i](group))
        features = torch.cat(features, dim=1)
        features = self.bn(features)
        features += x
        features = self.relu(features)
        return features

class MAF(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_dim=32, groups=4):
        """

        Args:
            sr_ch:
            seg_ch:
            hidden_dim:
            groups:
        """
        super(MAF, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, hidden_dim, 1, 1),
            SSC(hidden_dim, groups=groups)
        )
        self.sr_att = nn.Sequential(
            nn.Conv2d(hidden_dim, sr_ch, 1, 1),
            nn.Sigmoid()
        )
        self.seg_att = nn.Sequential(
            nn.Conv2d(hidden_dim, seg_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        cat = torch.cat([sr_fe, seg_fe], dim=1)
        fusion = self.fusion(cat)
        sr_att = self.sr_att(fusion)
        seg_att = self.seg_att(fusion)
        sr_out = sr_att*sr_fe + sr_fe
        seg_out = seg_att*seg_fe + seg_fe
        return sr_out, seg_out


class MAF1(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_dim=32):
        """

        Args:
            sr_ch:
            seg_ch:
            hidden_dim:
            groups:
        """
        super(MAF1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, hidden_dim, 1, 1),
            nn.ReLU()
        )
        self.conv_sp1 = nn.Conv2d(hidden_dim, hidden_dim,
                                  (7, 1), padding=(3, 0), bias=False)
        self.conv_sp2 = nn.Conv2d(hidden_dim, hidden_dim,
                                  (1, 7), padding=(0, 3), bias=False)

        self.fusion_seg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, seg_ch, 1, 1),
            nn.Sigmoid()
        )
        self.fusion_sr = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, sr_ch, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, sr_fe, seg_fe):
        cat = torch.cat([sr_fe, seg_fe], dim=1)
        fusion = self.conv1(cat)
        sp1 = self.conv_sp1(fusion)
        sp2 = self.conv_sp2(fusion)
        seg_fusion = self.fusion_seg(sp1 + sp2)
        sr_fusion = self.fusion_sr(sp1 + sp2)
        sr_out = sr_fusion*sr_fe + sr_fe
        seg_out = seg_fusion*seg_fe + seg_fe
        return sr_out, seg_out

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


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan=256):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
            ConvBNReLU(304, 256, ks=3, padding=1),
            ConvBNReLU(256, 256, ks=3, padding=1),
        )
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',
                                     align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        return feat_out
        # return self.conv_out(feat_out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class SRDecoder(nn.Module):
    def __init__(self, n_classes, low_chan=256):
        super(SRDecoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
            ConvBNReLU(304, 256, ks=3, padding=1),
            ConvBNReLU(256, 128, ks=3, padding=1),
        )
        self.conv_out = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',
                                     align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        return self.conv_out(feat_out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)



# class DeeplabV3Plus_maf(BaseModel):
#     def __init__(self, backbone='ResNet-50', class_num=19):
#         super(DeeplabV3Plus_maf, self).__init__(backbone=backbone)
#
#         self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
#         self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
#         self.head_top = UPerHead(in_channels=self.backbone.channels,
#                                  channel=32,
#                                  num_classes=2,
#                                  scales=(1, 2, 3, 6))
#
#         self.SRdecoder = SRDecoder(class_num, low_chan=self.backbone.channels[0])
#         self.init_weight()
#
#         self.pointwise = torch.nn.Sequential(
#             torch.nn.Conv2d(class_num, 3, 1),
#             torch.nn.BatchNorm2d(3),  # 添加了BN层
#             torch.nn.ReLU(inplace=True)
#         )
#
#         self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
#         self.up_edsr_1 = EDSRConv(64, 64)
#         self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.up_edsr_2 = EDSRConv(32, 32)
#         self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
#         self.up_edsr_3 = EDSRConv(16, 16)
#         self.up_conv_last = nn.Conv2d(16, 3, 1)
#
#         self.sr_seg_fusion_module = MAF(64, 256)
#         self.seg_out = nn.Conv2d(256, class_num, kernel_size=1, bias=False)
#         # self.sr_out = self.conv_out = nn.Conv2d(128, 64, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         H, W = x.size()[2:]
#         feat4, feat8, feat16, feat32 = self.backbone(x)
#         feat_aspp = self.aspp(feat32)
#         logits_bottom_seg = self.decoder(feat4, feat_aspp)
#         logits_bottom = self.seg_out(logits_bottom_seg)
#         logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)
#
#         if self.training:
#             logits_bottom = F.interpolate(logits_bottom, scale_factor=2, mode='bilinear', align_corners=True)
#
#             logits_sr = self.SRdecoder(feat4, feat_aspp)
#
#             logits_sr_up = self.up_sr_1(logits_sr)
#             logits_sr_up = self.up_edsr_1(logits_sr_up)
#             logits_sr_up = self.up_sr_2(logits_sr_up)
#             logits_sr_up = self.up_edsr_2(logits_sr_up)
#             logits_sr_up = self.up_sr_3(logits_sr_up)
#             logits_sr_up = self.up_edsr_3(logits_sr_up)
#             logits_sr_up = self.up_conv_last(logits_sr_up)
#
#             fusion_sr, fusion_seg = self.sr_seg_fusion_module(logits_sr, logits_bottom_seg)
#
#             fusion_seg = self.seg_out(fusion_seg)
#             fusion_seg = F.interpolate(fusion_seg, logits_bottom.size()[2:], mode="bilinear", align_corners=True)
#
#             fusion_sr = self.up_sr_1(fusion_sr)
#             fusion_sr = self.up_edsr_1(fusion_sr)
#             fusion_sr = self.up_sr_2(fusion_sr)
#             fusion_sr = self.up_edsr_2(fusion_sr)
#             fusion_sr = self.up_sr_3(fusion_sr)
#             fusion_sr = self.up_edsr_3(fusion_sr)
#             fusion_sr = self.up_conv_last(fusion_sr)
#
#             logits_top = self.head_top([feat4, feat8, feat16, feat32])
#             logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
#             logits_top = F.interpolate(logits_top, scale_factor=2, mode='bilinear', align_corners=True)
#             # return logits_bottom, logits_top, logits_sr_up, self.pointwise(logits_bottom), fusion_seg, None
#             return logits_bottom, logits_top, logits_sr_up, fusion_sr, fusion_seg, None
#         return logits_bottom
#
#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if ly.bias is not None:
#                     nn.init.constant_(ly.bias, 0)
#
#     def get_params(self):
#         back_bn_params, back_no_bn_params = self.backbone.get_params()
#         tune_wd_params = list(self.aspp.parameters()) + list(self.decoder.parameters()) + back_no_bn_params
#         no_tune_wd_params = back_bn_params
#         return tune_wd_params, no_tune_wd_params


class DeeplabV3Plus_maf(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19, upscale_rate=2):
        super(DeeplabV3Plus_maf, self).__init__(backbone=backbone)

        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))

        self.out_conv = nn.Conv2d(256, class_num, kernel_size=1, bias=False)
        self.init_weight()
        self.upscale_rate = upscale_rate

        self.sr = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.Conv2d(32, (upscale_rate ** 2) * 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=upscale_rate)
        )

        self.sr_seg_fusion_module = MAF1(256, 256)

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits_seg = self.decoder(feat4, feat_aspp)
        logits_bottom = self.out_conv(logits_seg)
        logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)

        if self.training:
            logits_bottom = F.interpolate(logits_bottom, scale_factor=self.upscale_rate, mode='bilinear', align_corners=True)

            logits_sr1 = self.decoder(feat4, feat_aspp)
            logits_sr = F.interpolate(logits_sr1, (H, W), mode="bilinear", align_corners=True)
            sr = self.sr(logits_sr)

            fusion_sr, fusion_seg = self.sr_seg_fusion_module(logits_sr1, logits_seg)

            fusion_seg = self.out_conv(fusion_seg)
            fusion_seg = F.interpolate(fusion_seg, logits_bottom.size()[2:], mode="bilinear", align_corners=True)

            fusion_sr = self.sr(fusion_sr)
            fusion_sr = F.interpolate(fusion_sr, logits_bottom.size()[2:], mode="bilinear", align_corners=True)

            logits_top = self.head_top([feat4, feat8, feat16, feat32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_top = F.interpolate(logits_top, scale_factor=self.upscale_rate, mode='bilinear',
                                          align_corners=True)
            return logits_bottom, logits_top, sr, fusion_seg, fusion_sr, None
        return logits_bottom

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.backbone.get_params()
        tune_wd_params = list(self.aspp.parameters()) + list(self.decoder.parameters()) + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


if __name__ == "__main__":
    from tqdm import tqdm
    net = DeeplabV3Plus_maf(backbone="MobileNetV3-large", class_num=8)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    # from semseg.utils.utils import count_parameters
    # print(f'model params cnt: {count_parameters(net)}MB')
    in_ten = torch.randn((2, 3, 512, 512)).cuda()
    _logits = net(in_ten)
    if net.training:
        print(_logits[0].shape, _logits[1].shape, _logits[2].shape, _logits[3].shape, _logits[4].shape)
    else:
        print(_logits.shape)
