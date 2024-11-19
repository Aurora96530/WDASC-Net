import torch
import torch.nn as nn
import torch.nn.functional as F


class NewClassifier(nn.Module):
    def __init__(self, num_classes, backbone_channels):
        super(NewClassifier, self).__init__()
        # 假设backbone_channels是一个包含backbone各层通道数的列表
        self.backbone = ...  # 使用您的backbone，但不要最后的分类层

        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 添加全连接层
        self.fc = nn.Linear(backbone_channels[-1], num_classes)

        # 初始化权重
        self.init_weight()

    def forward(self, x):
        # 通过backbone提取特征
        x = self.backbone(x)

        # 通过全局平均池化层
        x = self.global_avg_pool(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 通过全连接层得到分类结果
        x = self.fc(x)

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 假设您的分类任务有10个类别
num_classes = 10
model = NewClassifier(num_classes, backbone_channels=backbone.channels)

# 加载分割任务的预训练权重
pretrained_dict = torch.load('deeplabv3plus_pretrained.pth')
model_dict = model.state_dict()

# 过滤掉不需要加载的权重
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 冻结backbone的部分层
for param in model.backbone.parameters():
    param.requires_grad = False

# 训练模型...