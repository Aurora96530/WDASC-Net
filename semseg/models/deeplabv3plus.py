"""
@Project : semantic-segmentation
@File    : deeplabv3plus.py
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



def gram_matrix(features):
    # 假设features的形状是[batch_size, channels, height, width]
    b, c, h, w = features.size()
    # 展平height和width维度
    features = features.view(b, c, -1)  # 现在形状是[batch_size, channels, features]
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram

def normalize_gram(gram):
    # 如果gram是3维的，我们需要3个变量来解包
    b, c, f = gram.size()  # 这里的f是features的总数
    # 计算范数
    norm = torch.sqrt(torch.sum(gram ** 2, dim=2, keepdim=True) + 1e-6)  # 避免除零
    # 调整norm的形状以匹配gram的形状
    norm = norm.view(b, c, 1)  # 现在norm的形状是[batch_size, channels, 1]
    return gram / norm  # 执行归一化操作


def style_loss(gram_ref, gram_test):
    b, c, _ = gram_ref.shape
    loss = torch.mean((gram_ref - gram_test) ** 2)
    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B


def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape

    B, C, H, W = f_map1.shape
    HW = H * W

    if eye is None:
        eye = torch.eye(C).cuda()

    # feature map shape : (B,C,H,W) -> (B,C,HW)
    f_map1 = f_map1.contiguous().view(B, C, -1)
    f_map2 = f_map2.contiguous().view(B, C, -1)

    # f_cor shape : (B, C, C)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW - 1) + (eps * eye)

    return f_cor, B

def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape

    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).cuda()

    # get diagonal values of covariance matrix
    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).cuda()
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    diag_loss = diag_loss / B

    return diag_loss

def zca_whitening(features, epsilon=1e-6):
    # 计算特征的均值，并保持维度以便进行广播
    mean = features.mean(dim=(1, 2, 3), keepdim=True)
    features_centered = features - mean

    # 将四维特征张量转换为二维形式，每一行代表一个样本点，每一列代表一个特征
    batch_size, channels, height, width = features.size()
    features_reshaped = features_centered.permute(0, 3, 1, 2).reshape(batch_size * channels, -1)

    # 计算协方差矩阵
    cov = features_reshaped.t() @ features_reshaped / (batch_size * height * width - 1)

    # 计算协方差矩阵的特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(cov, UPLO='L')

    # 计算协方差矩阵的逆平方根
    inv_sqrt_eigvals = 1.0 / torch.sqrt(eigvals + epsilon)
    inv_sqrt_cov = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.t()

    # 应用ZCA白化变换
    features_whitened_reshaped = features_reshaped @ inv_sqrt_cov

    # 将白化后的特征恢复到原始形状
    features_whitened = features_whitened_reshaped.view(batch_size, channels, height, width)

    # 将均值加回到白化的特征上，注意均值张量的形状要匹配
    mean_to_add = mean.permute(0, 3, 1, 2)  # 保持 mean 的形状为 [batch_size, channels, 1, 1]
    features_zca = features_whitened + mean_to_add.expand_as(features_whitened)

    return features_zca

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


class ConvBNReLU_MSA(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU_MSA, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.msa = MSA(out_chan)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = self.msa(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ConvBNReLU_MWFC(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU_MWFC, self).__init__()
        # self.conv = nn.Conv2d(in_chan,
        #                       out_chan,
        #                       kernel_size=ks,
        #                       stride=stride,
        #                       padding=padding,
        #                       dilation=dilation,
        #                       bias=True)
        self.mwfc = MWFC(out_chan, out_chan, stride=1)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        # x = self.conv(x)
        x = self.mwfc(x)
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


class ASPP_dcam(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True):
        super(ASPP_dcam, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU_MSA(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU_MWFC(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU_MWFC(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU_MWFC(in_chan, out_chan, ks=3, dilation=18, padding=18)
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


class DeeplabV3Plus0(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19):
        super(DeeplabV3Plus0, self).__init__(backbone=backbone)

        # self.aspp = ASPP(in_chan=2048, out_chan=256, with_gp=True)
        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, _, _, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits = self.decoder(feat4, feat_aspp)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)

        return logits

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


class DeeplabV3Plus(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19):
        super(DeeplabV3Plus, self).__init__(backbone=backbone)

        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits_bottom = self.decoder(feat4, feat_aspp)
        logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)

        if self.training:
            logits_top = self.head_top([feat4, feat8, feat16, feat32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            # return torch.cat([logits_seg, logits_so], dim=1), logits_edge
            return logits_bottom, logits_top, None
            # return logits_bottom, logits_top, feat32, None
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


class DeeplabV3Plus1(BaseModel):
    def __init__(self, backbone='ResNet-50', class_num=19):
        super(DeeplabV3Plus1, self).__init__(backbone=backbone)

        self.aspp = ASPP(in_chan=self.backbone.channels[-1], out_chan=256, with_gp=True)
        self.decoder = Decoder(class_num, low_chan=self.backbone.channels[0])
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits_bottom = self.decoder(feat4, feat_aspp)
        logits_bottom = F.interpolate(logits_bottom, (H, W), mode='bilinear', align_corners=True)

        if self.training:
            logits_top = self.head_top([feat4, feat8, feat16, feat32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            # return torch.cat([logits_seg, logits_so], dim=1), logits_edge
            return logits_bottom, logits_top, feat32, None
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
    from thop import profile
    from torch.cuda import Event

    # net = DeeplabV3Plus(backbone="MobileNetV3-large", class_num=8)
    # net.init_pretrained('/home_lv/jiaoli.liu/sosnet/checkpoints/backbones/mobilenet/mobilenetv3_large.pth')
    # net.cuda()
    # net.train()
    # net = nn.DataParallel(net)
    # from semseg.utils.utils import count_parameters
    # print(f'model params cnt: {count_parameters(net)}MB')
    # in_ten = torch.randn((2, 3, 720, 960)).cuda()
    # _logits = net(in_ten)
    #     if net.training:
    #         print(_logits[0].shape, _logits[1].shape)
    #     else:
    #         print(_logits.shape)

    net = DeeplabV3Plus(backbone="MobileNetV2-", class_num=5)
    net.init_pretrained('/home_lv/jiaoli.liu/sosnet/checkpoints/backbones/mobilenet/mobilenet_v2.pth')
    net = net.eval().cuda()
    in_ten = torch.randn((1, 3, 512, 512)).cuda()
    flops, params = profile(net, inputs=(in_ten, ))

    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    inference_times = []

    with torch.no_grad():
        for _ in range(10):
            net(in_ten)
    # start_event.record()
    for _ in range(400):
        start_event.record()
        with torch.no_grad():
            output = net(in_ten)
        end_event.record()

        torch.cuda.synchronize()
        inference_times.append(start_event.elapsed_time(end_event))

    # 打印结果
    print(f"Inference Time: {sum(inference_times) / len(inference_times):.2f} s")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为GFLOPs
    print(f"Params: {params / 1e6:.2f} M")    # 转换为百万（M）

