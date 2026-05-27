import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def create_gaussian_kernel(size, sigma):
    """ 创建一个3D高斯核 """
    kernel = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2
    sum_val = 0.0

    for x in range(size):
        for y in range(size):
            for z in range(size):
                x_dist = x - center
                y_dist = y - center
                z_dist = z - center
                kernel[x, y, z] = np.exp(-(x_dist**2 + y_dist**2 + z_dist**2) / (2 * sigma**2))
                sum_val += kernel[x, y, z]

    kernel /= sum_val
    return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def apply_gaussian_blur(tensor,size,sigma):
    """ 在3D张量上应用高斯模糊 """
    # 需要确保输入张量是float类型
    tensor = tensor.float()
    gaussian_kernel = create_gaussian_kernel(size, sigma)

    # 定义卷积层
    conv = F.conv3d
    padding = gaussian_kernel.shape[2] // 2
    kernel = gaussian_kernel.to(tensor.device)  # 将核移动到与输入相同的设备
    blurred_tensor = conv(tensor, kernel, padding=padding)

    return blurred_tensor

# tensor = torch.randn(1, 1, 300, 200, 300).to('cuda')  # 示例输入张量
# # tensor_np=tensor.numpy()
# # plt.figure(figsize=(12, 6))
# # plt.imshow(tensor_np[0,0,200])
# # plt.show()
#
# blurred_tensor = apply_gaussian_blur(tensor,9,5)
#
# blurred_tensor_np=blurred_tensor.to('cpu').numpy()
# print(blurred_tensor_np.shape)
# # plt.figure(figsize=(12, 6))
# # plt.imshow(blurred_tensor[0,0,200])
# # plt.show()