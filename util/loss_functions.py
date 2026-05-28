import math
import random
import time

# import unfold
# import util.unfold
import numpy as np
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable


class smoothing_loss(nn.Module):
    def __init__(self):
        super(smoothing_loss, self).__init__()

    def forward(self, y_pred):
        # 计算 x, y, z 方向的梯度
        dx = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        dy = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        dz = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
        # 处理尺寸不匹配的问题
        dx = dx[:, :, :, :- 1, : - 1]
        dy = dy[:, :, :- 1, :, :- 1]
        dz = dz[:, :, : - 1, : - 1, :]
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-8)  # 添加小值防止除零
        # gradient_magnitude = dx ** 2 + dy ** 2 + dz ** 2

        # 计算所有点的梯度模长的平均值
        loss = gradient_magnitude.mean()

        return loss


class bias_mean_loss(nn.Module):
    def __init__(self):
        super(bias_mean_loss, self).__init__()

    def forward(self, bias, mask_w, mask_f):
        # 1. 水区域均值损失
        mean_w = torch.sum(bias * mask_w) / (torch.sum(mask_w) + 1e-8)
        loss_w = torch.abs(1 - mean_w)

        # 2. 脂肪区域均值损失
        mean_f = torch.sum(bias * mask_f) / (torch.sum(mask_f) + 1e-8)
        loss_f = torch.abs(1 - mean_f)

        # 3. 水脂以外的区域 (Background/Others)
        # 假设 mask 值为 0 或 1，1-mask 就是以外的区域
        mask_others = 1.0 - torch.clamp(mask_w + mask_f, 0, 1)
        mean_others = torch.sum(bias * mask_others) / (torch.sum(mask_others) + 1e-8)
        # 对于以外区域，通常约束它趋向 1 也是一种正则化手段
        loss_others = torch.abs(1 - mean_others)
        return (loss_w + loss_f) + loss_others
# import torch
# import torch.nn as nn
# class bias_mean_loss(nn.Module):
#     def __init__(self, eps=1e-8, use_l2=True):
#         super(bias_mean_loss, self).__init__()
#         self.eps = eps
#         self.use_l2 = use_l2
#     def masked_log_mean(self, log_bias, mask):
#         return torch.sum(log_bias * mask) / (torch.sum(mask) + self.eps)
#     def penalty(self, x):
#         if self.use_l2:
#             return x ** 2
#         else:
#             return torch.abs(x)
#     def forward(self, bias, mask_w, mask_f):
#         bias = torch.clamp(bias, min=self.eps)
#         log_bias = torch.log(bias)
#         mean_log_w = self.masked_log_mean(log_bias, mask_w)
#         mean_log_f = self.masked_log_mean(log_bias, mask_f)
#         mask_others = 1.0 - torch.clamp(mask_w + mask_f, 0, 1)
#         mean_log_others = self.masked_log_mean(log_bias, mask_others)
#         loss_w = self.penalty(mean_log_w)
#         loss_f = self.penalty(mean_log_f)
#         loss_others = self.penalty(mean_log_others)
#         return (loss_w + loss_f + loss_others)

class SegLoss(torch.nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, a, b, ts, wpeak):
        a = a * wpeak

        b = b

        c = a / b
        ts = torch.where(ts > 0, True, False)
        c_t = torch.masked_select(c, ts)

        c_std = torch.std(c_t)
        c_mean = torch.mean(c_t)
        cor = c_std / c_mean

        return cor

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RetinexLoss(nn.Module):
    def __init__(self, scales=[1, 2, 3], eps=1e-6):
        super(RetinexLoss, self).__init__()
        self.scales = scales
        self.eps = eps

    def get_gaussian_kernel(self, sigma, truncate=0.5, device='cuda'):
        """生成 1D 高斯核，适配 3D 卷积"""
        radius = int(math.ceil(sigma * truncate))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum() + self.eps  # 防止除零
        return kernel.view(1, 1, -1, 1, 1)  # 形状 (1, 1, k, 1, 1)

    def separable_gaussian_3d(self, x, sigma):
        """分离卷积实现 3D 高斯模糊"""
        channels = x.size(1)
        kernel_1d = self.get_gaussian_kernel(sigma, device=x.device)
        padding = kernel_1d.size(2) // 2  # 计算填充大小

        # 调整核方向以适配 x, y, z 卷积
        kernel_x = kernel_1d  # (1, 1, k, 1, 1)
        kernel_y = kernel_1d.permute(0, 1, 3, 2, 4)  # (1, 1, 1, k, 1)
        kernel_z = kernel_1d.permute(0, 1, 4, 3, 2)  # (1, 1, 1, 1, k)

        # 分别对 x, y, z 方向卷积，groups=channels 确保通道独立
        x = F.conv3d(x, kernel_x.expand(channels, -1, -1, -1, -1), padding=(padding, 0, 0), groups=channels)
        x = F.conv3d(x, kernel_y.expand(channels, -1, -1, -1, -1), padding=(0, padding, 0), groups=channels)
        x = F.conv3d(x, kernel_z.expand(channels, -1, -1, -1, -1), padding=(0, 0, padding), groups=channels)
        return x

    def msr_3d(self, image):
        """多尺度 Retinex"""
        image = torch.clamp(image, min=0)  # 确保输入非负
        retinex = torch.zeros_like(image)
        for scale in self.scales:
            sigma = scale / 2.0
            blurred = self.separable_gaussian_3d(image, sigma)
            retinex += torch.log(image + self.eps) - torch.log(blurred + self.eps)
        retinex = retinex / len(self.scales)  # 限制范围防止溢出
        return torch.exp(retinex)

    def forward(self, input, target, mask, wPeak):

        input = input * wPeak
        target = target * wPeak

        retinex_input = self.msr_3d(input)*10
        retinex_target = self.msr_3d(target)*10

        # 应用掩码
        retinex_input = torch.where(mask == 0, 0, retinex_input)
        retinex_target = torch.where(mask == 0, 0, retinex_target)

        loss = torch.mean(torch.square(retinex_target - retinex_input)[mask == 1])
        return loss

