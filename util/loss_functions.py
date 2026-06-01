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

