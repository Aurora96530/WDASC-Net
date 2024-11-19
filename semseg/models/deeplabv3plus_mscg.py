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


class DownAttention4D(nn.Module):
    def __init__(self, dim, down_rate=32, num_head=4, sclae_init=1e-6) -> None:
        super().__init__()
        self.down_conv = nn.Conv2d(dim, dim, kernel_size=down_rate, stride=down_rate, groups=dim)
        self.head_dim = dim // num_head
        self.num_head = num_head
        self.gamma = self.head_dim ** -0.5
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.layer_scaler = nn.Parameter(torch.ones(dim) * sclae_init, requires_grad=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, 1),
            nn.GELU()
        )
        self.ffn_scaler = nn.Parameter(torch.ones(dim) * sclae_init, requires_grad=True)

    def forward(self, x):
        down_x = self.down_conv(x)
        bs, ch, h, w = down_x.size()
        q_x = self.q_proj(down_x).reshape(bs, self.num_head, self.head_dim, h * w).transpose(2,
                                                                                             3)  # bs, num_head, N, head_num
        k_x = self.k_proj(down_x).reshape(bs, self.num_head, self.head_dim, h * w)  # bs, num_head, head_num, N
        v_x = self.v_proj(down_x).reshape(bs, self.num_head, self.head_dim, h * w).transpose(2,
                                                                                             3)  # bs, num_head, N, head_num
        att = q_x @ k_x
        att = torch.softmax(att * self.gamma, dim=-1)
        net = att @ v_x
        net = net.transpose(2, 3).reshape(bs, ch, h, w)
        net = self.layer_scaler.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * net + down_x
        net = self.ffn(net) * self.ffn_scaler.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + net
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=True)
        return net


class CAGL(nn.Module):
    def __init__(self, in_ch1, in_ch2, hidden_state=32, reduction=4, down_rate=32, num_head=4) -> None:
        super(CAGL, self).__init__()
        fu_ch = in_ch1 + in_ch2
        self.fu_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fu_ch, fu_ch // reduction, 1, 1),
            nn.ReLU(),
            nn.Conv2d(fu_ch // reduction, fu_ch, 1, 1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(fu_ch, hidden_state, 1, 1)
        self.global_att = DownAttention4D(hidden_state, down_rate=down_rate, num_head=num_head)
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, 3, 1, 1, groups=hidden_state),
            nn.GELU(),
            nn.Conv2d(hidden_state, hidden_state, 1, 1),
            nn.GELU()
        )
        self.out_x = nn.Sequential(
            nn.Conv2d(hidden_state, in_ch1, 1, 1),
            nn.Softmax(dim=1)
        )
        self.out_y = nn.Sequential(
            nn.Conv2d(hidden_state, in_ch2, 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        fuse = torch.cat([x, y], dim=1)
        fuse = self.fu_se(fuse) * fuse
        fuse = self.fuse(fuse)
        fuse_g = self.global_att(fuse)
        fuse_l = self.local_conv(fuse)
        fuse = fuse + fuse_g + fuse_l
        o_x = self.out_x(fuse) * x + x
        o_y = self.out_y(fuse) * y + y
        return o_x, o_y


class SplitSpatialConv(nn.Module):
    def __init__(self, ch, cards):
        super(SplitSpatialConv, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(cards):
            self.convs.append(
                nn.Conv2d(ch, ch, 3, 1, padding=i+1, dilation=i+1, groups=ch, bias=False)
            )
        # self.convs.append(
        #     ImagePool(ch)
        # )
        self.fusion = nn.Conv2d(ch*cards, ch, 1, 1, 0)
    def forward(self, x):
        nets = []
        for conv in self.convs:
            nets.append(conv(x))
        return self.fusion(torch.cat(nets, dim=1))


class CrossAttentionConv(nn.Module):
    def __init__(self, x_ch, y_ch, dim=64):
        super(CrossAttentionConv, self).__init__()
        self.x_map_conv = nn.Sequential(
            nn.Conv2d(x_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.y_map_conv = nn.Sequential(
            nn.Conv2d(y_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        #spatial
        self.x_spatial = nn.Sequential(
            SplitSpatialConv(2, cards=4),
            nn.Conv2d(2, 1, 1, 1, bias=False),
            # nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )

        self.y_spatial = nn.Sequential(
            SplitSpatialConv(2, cards=4),
            nn.Conv2d(2, 1, 1, 1, bias=False),
            # nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )
        #channel
        self.x_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, 1),
            nn.Conv2d(dim//4, dim, 1, 1),
            nn.Sigmoid()
        )

        self.y_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, 1),
            nn.Conv2d(dim // 4, dim, 1, 1),
            nn.Sigmoid()
        )

        #
        self.x_out = nn.Conv2d(dim, x_ch, 1, 1)
        self.y_out = nn.Conv2d(dim, y_ch, 1, 1)

    def forward(self, fes):
        x, y = fes
        x_hidden = self.x_map_conv(x)
        y_hidden = self.y_map_conv(y)

        #channel
        x_channel = self.x_channel(x_hidden)
        y_channel = self.y_channel(y_hidden)
        x_hidden = y_channel * x_hidden
        y_hidden = x_channel * y_hidden
        #spatial
        x_max = torch.max(x_hidden, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x_hidden, dim=1, keepdim=True)
        x_spatial = torch.cat([x_max, x_avg], dim=1)

        x_spatial = self.x_spatial(x_spatial)

        y_max = torch.max(y_hidden, dim=1, keepdim=True)[0]
        y_avg = torch.mean(y_hidden, dim=1, keepdim=True)
        y_spatial = torch.cat([y_max, y_avg], dim=1)
        y_spatial = self.y_spatial(y_spatial)
        x_hidden = x_hidden * y_spatial
        y_hidden = y_hidden * x_spatial

        x = self.x_out(x_hidden) + x
        y = self.y_out(y_hidden) + y
        return x, y


class CrossTaskAttention(nn.Module):
    def __init__(self, x_ch, y_ch, dim=32, num_head=8, qkv_bias=False, patch_size=32):
        super(CrossTaskAttention, self).__init__()
        self.cross_attention = CrossAttentionConv(x_ch, y_ch, dim)

    def forward(self, x, y):
        x, y = self.cross_attention([x, y])
        return x, y


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
        # self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

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


class DeeplabV3Plus_mscg(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19, upscale_rate=2):
        super(DeeplabV3Plus_mscg, self).__init__(backbone=backbone)

        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))

        self.SRdecoder = SRDecoder(class_num, low_chan=self.backbone.channels[0])
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

        # self.sr_seg_fusion_module = CrossTaskAttention(256, 256, patch_size=16)
        self.sr_seg_fusion_module = CAGL(256, 256)

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits_seg = self.decoder(feat4, feat_aspp)
        logits_bottom = self.out_conv(logits_seg)
        logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)

        if self.training:
            logits_bottom = F.interpolate(logits_bottom, scale_factor=self.upscale_rate, mode='bilinear', align_corners=True)

            logits_sr = self.decoder(feat4, feat_aspp)
            fusion_sr, fusion_seg = self.sr_seg_fusion_module(logits_sr, logits_seg)
            fusion_seg = self.out_conv(fusion_seg)
            fusion_seg = F.interpolate(fusion_seg, logits_bottom.size()[2:], mode="bilinear", align_corners=True)

            logits_sr = F.interpolate(logits_sr, (H, W), mode="bilinear", align_corners=True)
            sr = self.sr(logits_sr)

            fusion_sr = F.interpolate(fusion_sr, (H, W), mode="bilinear", align_corners=True)
            fusion_sr = self.sr(fusion_sr)

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
    net = DeeplabV3Plus_mscg(backbone="MobileNetV3-large", class_num=8)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    # from semseg.utils.utils import count_parameters
    # print(f'model params cnt: {count_parameters(net)}MB')
    in_ten = torch.randn((2, 3, 256, 256)).cuda()
    _logits = net(in_ten)
    if net.training:
        print(_logits[0].shape, _logits[1].shape, _logits[2].shape, _logits[3].shape, _logits[4].shape)
    else:
        print(_logits.shape)
