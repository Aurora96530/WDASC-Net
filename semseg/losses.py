import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None,
                 aux_weights: [tuple, list] = (1, 0.4, 0.4)) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7,
                 aux_weights: [tuple, list] = (1, 1)) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 5
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


# class SOEM(nn.Module):
#
#     def __init__(self, ignore_label=255, ratio=0.1, threshold=0.5) -> None:
#         """
#         Small object example mining for SOSNet
#         Args:
#             ignore_label: int, ignore label id in dataset
#             ratio:
#             threshold:
#         """
#         super().__init__()
#         self.ignore_label = ignore_label
#         self.ratio = ratio
#         self.threshold = threshold
#
#     def forward(self, loss: Tensor, labels: Tensor, labels_s: Tensor) -> Tensor:
#         """
#         Args:
#             loss: the joint loss, 0 where the ground truth label is ignored.
#             labels: the segmentation labels
#             labels_s: the small objet labels that indicate where the small objects are.
#
#         Returns:
#             loss_hard: the mean value of those hardest mse losses.
#         """
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         n_min = int(labels[labels != self.ignore_label].numel() * self.ratio)
#         loss_flat = loss.contiguous().view(-1)
#         labels_s_flat = labels_s.contiguous().view(-1)
#         loss_s = loss_flat[labels_s_flat == 1]
#         loss_l = loss_flat[labels_s_flat == 0]
#         loss_hard_s = loss_s[loss_s > self.threshold]
#         loss_hard_l = loss_l[loss_l > self.threshold]
#
#         if loss_hard_s.numel() < n_min:
#             if loss_s.numel() <= n_min:
#                 loss_hard_s = loss_s
#             else:
#                 loss_hard_s, _ = loss_s.topk(n_min)
#
#         if loss_hard_l.numel() < n_min:
#             if loss_l.numel() <= n_min:
#                 loss_hard_l = loss_l
#             else:
#                 loss_hard_l, _ = loss_l.topk(n_min)
#
#         loss_hard = (torch.sum(loss_hard_s) + torch.sum(loss_hard_l)) / (loss_hard_s.numel() + loss_hard_l.numel())
#
#         # return torch.mean(loss)
#         return loss_hard


class BinaryDiceLoss(torch.nn.Module):

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, logits, targets):
        """
        Dice loss for binary segmentation.
        Note that the logits can't be activated before calculating this loss.
        Args:
            logits: torch.FloatTensor, predicted probabilities without sigmoid, shape=(n_batch, h, w)
            targets: torch.LongTensor, ground truth probabilities, shape=(n_batch, h, w)
        Returns:
            score: torch.FloatTensor, dice loss, shape=(1,)
        """
        num = targets.size(0)  # batch size
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: [tuple, list] = (1, 0.4, 0.4)):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels * preds, dim=(2, 3))
        fn = torch.sum(labels * (1 - preds), dim=(2, 3))
        fp = torch.sum((1 - labels) * preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class Focal(torch.nn.Module):
    def __init__(self, ignore_index=255, weight=None, gamma=2, alpha=None, reduction='none'):
        super(Focal, self).__init__()
        self.ignore_label = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha]).cuda()
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        self.reduction = reduction

    def forward(self, logits, target):
        bs, h, w = target.shape
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = torch.log_softmax(logits, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.reshape(-1, h, w)
        target = target.reshape(-1, h, w)
        mean_loss = loss.mean().detach()
        loss = torch.where(target != self.ignore_label, loss, mean_loss)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# class Focal(nn.Module):
#     def __init__(self, ignore_index=255, weight=None, gamma=2, size_average=True):
#         super(Focal, self).__init__()
#         self.gamma = gamma
#         self.size_average = size_average
#         self.CE_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
#
#     def forward(self, output, target):
#         logpt = self.CE_loss(output, target)
#         pt = torch.exp(-logpt)
#         loss = ((1 - pt) ** self.gamma) * logpt
#         return loss
#         # if self.size_average:
#         #     return loss.mean()
#         # return loss.sum()


class FocalDice(nn.Module):
    def __init__(self, ignore_index=255, weight=None, gamma=2, size_average=True,
                 delta: float = 0.5, aux_weights: [tuple, list] = (1, 0.4, 0.4)):
        super(FocalDice, self).__init__()
        self.focal = Focal(ignore_index, weight, gamma, size_average)
        self.dice = Dice(delta, aux_weights)

    def forward(self, output, target):
        return self.focal(output, target) + self.dice(output, target)


class DiceBCELoss(nn.Module):

    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = BinaryDiceLoss()
        self.bce = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, targets):
        """
        A loss combine binary dice loss and binary cross-entropy loss.
        Note that the logits can't be activated before calculating this loss.
        Args:
            logits: torch.FloatTensor, predicted probabilities without sigmoid, shape=(n_batch, h, w)
            targets: torch.LongTensor, ground truth probabilities, shape=(n_batch, h, w)
        Returns:
            loss_diceBce
        """
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets)
        loss_diceBce = loss_dice + loss_bce
        return loss_diceBce


class FALoss(torch.nn.Module):
    def __init__(self, subscale=0.0625):
        super(FALoss, self).__init__()
        self.subscale = int(1/subscale)

    def forward(self, feature1, feature2):
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2) #[N,W*H,W*H]

        L1norm=torch.norm(mat2-mat1, 1)

        return L1norm/((height*width)**2)


class FALoss1(torch.nn.Module):
    def __init__(self, subscale=0.0625):
        super(FALoss1, self).__init__()
        self.subscale = int(1 / subscale)

    def forward(self, feature1, feature2):
        # 对特征图进行平均池化
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        # 将特征图展平为二维张量
        feature1 = feature1.view(feature1.size(0), feature1.size(1), -1)  # shape: (N, C, H*W)
        feature2 = feature2.view(feature2.size(0), feature2.size(1), -1)  # shape: (N, C, H*W)

        # 计算相关系数
        corr_coef = torch.zeros(feature1.size(1))
        for i in range(feature1.size(1)):
            # 计算每个通道之间的相关系数
            cov = torch.mean((feature1[:, i] - torch.mean(feature1[:, i])) * (feature2[:, i] - torch.mean(feature2[:, i])))
            std_f1 = torch.std(feature1[:, i])
            std_f2 = torch.std(feature2[:, i])
            corr_coef[i] = cov / (std_f1 * std_f2 + 1e-8)  # 避免除以零

        # 将相关系数转换为相似性度量，并求平均值作为最终损失
        similarity = 1 - torch.mean(torch.abs(corr_coef))

        return similarity

from pytorch_msssim import SSIM

class SSIMLoss(nn.Module):
    def __init__(self,
                data_range=255,
                size_average=True,
                win_size=11,
                win_sigma=1.5,
                channel=3,
                spatial_dims=2,
                K=(0.01, 0.03),
                nonnegative_ssim=False,):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=data_range, win_size=win_size, size_average=size_average,
                         win_sigma=win_sigma, channel=channel, spatial_dims=spatial_dims, K=K,
                         nonnegative_ssim=nonnegative_ssim)

    def forward(self, sr, hr):
        ssim = self.ssim(sr, hr)
        loss = 1 - ssim
        return loss

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel

def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
    """get map pairs
    Args:
        labels_4D	:	labels, shape [N, C, H, W]
        probs_4D	:	probabilities, shape [N, C, H, W]
        radius		:	the square radius
        Return:
            tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
    """
    # pad to ensure the following slice operation is valid
    # pad_beg = int(radius // 2)
    # pad_end = radius - pad_beg

    # the original height and width
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    # https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
    # padding = (pad_beg, pad_end, pad_beg, pad_end)
    # labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

    # get the neighbors
    la_ns = []
    pr_ns = []
    # for x in range(0, radius, 1):
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        # for calculating RMI
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        # for other purpose
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors

def log_det_by_cholesky(matrix):
    """
    Args:
        matrix: matrix must be a positive define matrix.shape [N, C, D, D].
    Ref:
       https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
    """
    # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
    # where C is the cholesky decomposition of A.
    chol = torch.linalg.cholesky(matrix)
    # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

def map_get_pairs_region(labels_4D, probs_4D, radius=3, is_combine=0, num_classeses=21):
    """get map pairs
    Args:
        labels_4D	:	labels, shape [N, C, H, W].
        probs_4D	:	probabilities, shape [N, C, H, W].
        radius		:	The side length of the square region.
        Return:
            A tensor with shape [N, C, radiu * radius, H // radius, W // raidius]
    """
    kernel = torch.zeros([num_classeses, 1, radius, radius]).type_as(probs_4D)
    padding = radius // 2
    # get the neighbours
    la_ns = []
    pr_ns = []
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            kernel_now = kernel.clone()
            kernel_now[:, :, y, x] = 1.0
            la_now = F.conv2d(labels_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
            pr_now = F.conv2d(probs_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        # for calculating RMI
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        # for other purpose
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors
    return

def batch_cholesky_inverse(matrix):
    """
    Args:
        matrix, 4-D tensor, [N, C, M, M]. matrix must be a symmetric positive define matrix.
    """
    chol_low = torch.linalg.cholesky(matrix, upper=False)
    chol_low_inv = batch_low_tri_inv(chol_low)
    return torch.matmul(chol_low_inv.transpose(-2, -1), chol_low_inv)

def batch_low_tri_inv(L):
    """
    Batched inverse of lower triangular matrices
    Args:
        L :	a lower triangular matrix
    Ref:
        https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing
    """
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / L[..., j, j]
        for i in range(j + 1, n):
            S = 0.0
            for k in range(0, i + 1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / L[..., i, i]
    return invL

class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        # loss = self.forward_softmax_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):
        """
        Using both softmax and sigmoid operations.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=_CLIP_MIN, max=_CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D 	:	[N, C, H, W], dtype=float32
            probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        # #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'Focal', 'FocalDice', 'DiceBCELoss', 'SSIMLoss', 'RMILoss',
           'get_loss', 'FALoss']
# __all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'Focal', 'FocalDice', 'DiceBCELoss', 'get_loss']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\n" \
                                    f"Available loss functions: {__all__} "
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    _pred = torch.randint(0, 2, (2, 3, 480, 640), dtype=torch.float).cuda()
    _label = torch.randint(0, 3, (2, 480, 640), dtype=torch.long).cuda()
    _pred2 = torch.randint(0, 2, (2, 480, 640), dtype=torch.float).cuda()
    _label2 = torch.randint(0, 2, (2, 480, 640), dtype=torch.float).cuda()
    loss_fn = RMILoss()
    # loss_fn = Focal(ignore_index=0)
    y = loss_fn(_pred, _label)
    print(y)
