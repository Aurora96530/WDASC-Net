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
        return self.conv_out(feat_out)

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


class DeeplabV3Plus_mfa(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19):
        super(DeeplabV3Plus_mfa, self).__init__(backbone=backbone)

        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))

        self.SRdecoder = SRDecoder(class_num, low_chan=self.backbone.channels[0])
        self.init_weight()

        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(class_num, 3, 1),
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

        # self.query = FSL(3, class_num)


    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits_bottom = self.decoder(feat4, feat_aspp)
        logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)

        if self.training:
            logits_bottom = F.interpolate(logits_bottom, scale_factor=2, mode='bilinear', align_corners=True)

            logits_sr = self.SRdecoder(feat4, feat_aspp)

            logits_sr_up = self.up_sr_1(logits_sr)
            logits_sr_up = self.up_edsr_1(logits_sr_up)
            logits_sr_up = self.up_sr_2(logits_sr_up)
            logits_sr_up = self.up_edsr_2(logits_sr_up)
            logits_sr_up = self.up_sr_3(logits_sr_up)
            logits_sr_up = self.up_edsr_3(logits_sr_up)
            logits_sr_up = self.up_conv_last(logits_sr_up)

            # seg_weight = self.query(logits_sr_up, logits_bottom)
            # fusion_seg = seg_weight * logits_bottom + logits_bottom

            logits_top = self.head_top([feat4, feat8, feat16, feat32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_top = F.interpolate(logits_top, scale_factor=2, mode='bilinear', align_corners=True)

            return logits_bottom, logits_top, logits_sr_up
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
    net = DeeplabV3Plus_mfa(backbone="MobileNetV3-large", class_num=8)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    # from semseg.utils.utils import count_parameters
    # print(f'model params cnt: {count_parameters(net)}MB')
    in_ten = torch.randn((2, 3, 512, 512)).cuda()
    _logits = net(in_ten)
    if net.training:
        print(_logits[0].shape, _logits[1].shape, _logits[2].shape, _logits[3].shape)
    else:
        print(_logits.shape)
