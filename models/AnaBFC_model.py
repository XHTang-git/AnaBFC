import random

import numpy as np
import torch
from collections import OrderedDict

from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks3d as networks
from util.loss_functions import bias_mean_loss, smoothing_loss, SegLoss
import visdom
import math
import torch.nn.functional as F

vis = visdom.Visdom(env="WFsperate", port=8097)

images = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'U_pre'})
images1 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'ori'})
images2 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'N4'})
images3 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'B'})
images4 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'Bpre'})
images5 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'W'})
images6 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'F'})
images7 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'M'})
images8 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'TissueNorm'})
images9 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'uniform_template'})
images10 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'uniform_template1'})
images11 = vis.images(np.zeros((3, 1, 256, 192)), opts={'title': 'V_gen'})


def visShow(data, img,title):
    data = data.detach().cpu().numpy()
    depth = data.shape[2]
    imageVis = []
    for m in range(0, data.shape[1], 1):
        visd = []
        for s in range(0, depth, int((depth-10)/4)):
            # print(s)
            slice = data[0, m, s, :, :]
            slice=np.where(slice>0,slice/np.max(slice)*255,0)
            visd.append(slice)
        imageVis.append(np.vstack(visd))
    imageVis = np.concatenate(imageVis, axis=1)
    imageVis = imageVis.reshape([1, imageVis.shape[0], imageVis.shape[1]])
    vis.images(imageVis, win=img,opts={'title':title,'width': 300,'height':1000})

class AnaBFC(BaseModel):
    def name(self):
        return 'AnaBFC'

    def _to_bool(self, x):
        if isinstance(x, bool):
            return x
        if torch.is_tensor(x):
            if x.numel() == 1:
                return bool(x.item())
            return bool(x[0].item())
        if isinstance(x, (list, tuple)):
            return bool(x[0])
        return bool(x)

    def _is_in_semi_phase(self):
        print(self.i)
        return self.semi_supervised and (self.i >= self.semi_warmup_iters)

    def _gaussian_kernel_3d(self, kernel_size=5, sigma=1.0, channels=1, device='cuda'):
        ax = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1, 1)
        return kernel

    def _smooth_bias(self, bias, sigma=1.0, kernel_size=5):
        channels = bias.shape[1]
        kernel = self._gaussian_kernel_3d(
            kernel_size=kernel_size,
            sigma=sigma,
            channels=channels,
            device=bias.device
        )
        padding = kernel_size // 2
        bias_smooth = F.conv3d(bias, kernel, padding=padding, groups=channels)
        return bias_smooth

    def _affine_transform_bias(self, bias, max_rot_deg=5.0, max_shift=4):
        """
        bias: [B, C, D, H, W]
        return: affine transformed bias, same shape
        """
        B, C, D, H, W = bias.shape
        device = bias.device
        dtype = bias.dtype
        theta_all = []
        for _ in range(B):
            # small rotations around x/y/z
            rx = math.radians(random.uniform(-max_rot_deg, max_rot_deg))
            ry = math.radians(random.uniform(-max_rot_deg, max_rot_deg))
            rz = math.radians(random.uniform(-max_rot_deg, max_rot_deg))
            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)
            Rx = torch.tensor([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ], device=device, dtype=dtype)
            Ry = torch.tensor([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], device=device, dtype=dtype)
            Rz = torch.tensor([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ], device=device, dtype=dtype)
            # rotation order: Rz @ Ry @ Rx
            R = torch.matmul(Rz, torch.matmul(Ry, Rx))
            # translation in normalized coordinates [-1, 1]
            td = random.uniform(-max_shift, max_shift) * 2.0 / max(D - 1, 1)
            th = random.uniform(-max_shift, max_shift) * 2.0 / max(H - 1, 1)
            tw = random.uniform(-max_shift, max_shift) * 2.0 / max(W - 1, 1)
            theta = torch.zeros((3, 4), device=device, dtype=dtype)
            theta[:, :3] = R
            # grid_sample for 5D uses x,y,z order => W,H,D
            theta[0, 3] = tw
            theta[1, 3] = th
            theta[2, 3] = td
            theta_all.append(theta)
        theta_all = torch.stack(theta_all, dim=0)  # [B, 3, 4]
        grid = F.affine_grid(theta_all, size=bias.size(), align_corners=True)
        bias_affine = F.grid_sample(
            bias,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return bias_affine

    def _intensity_transform_bias(self, detached_fake_B, low_range=(1500, 8000)):
        """
        detached_fake_B: [B, C, D, H, W]
        return: intensity-transformed bias
        """
        bf_Min = torch.min(detached_fake_B)
        l = np.log10(low_range[0]) * 10000
        h = np.log10(low_range[1]) * 10000
        low = random.randint(int(l), int(h))
        low = low / 10000
        low = 10 ** low / 10000
        detached_fake_B_aug = torch.where(
            detached_fake_B < 1,
            low + (detached_fake_B - bf_Min) / (1 - bf_Min + 1e-8) * (1 - low),
            detached_fake_B
        )
        detached_fake_B_aug = torch.where(
            detached_fake_B_aug > 1,
            1 + (detached_fake_B_aug - 1) * (1 - low) / (1 - bf_Min + 1e-8),
            detached_fake_B_aug
        )
        return detached_fake_B_aug

    def augment_bias_field(self,
                           bias,
                           max_rot_deg=5.0,
                           max_shift=4,
                           low_range=(1500, 8000),
                           sigma_range=(0.8, 1.5),
                           kernel_size=5,
                           clip_min=0.03,
                           clip_max=5.0):
        """
        Full bias augmentation:
        1) affine + translation
        2) intensity transform
        3) gaussian smoothing
        bias: [B, C, D, H, W]
        return: augmented bias field
        """
        bias_aug = bias
        # 1. affine + translation
        bias_aug = self._affine_transform_bias(
            bias_aug,
            max_rot_deg=max_rot_deg,
            max_shift=max_shift
        )
        # 2. intensity transform (your current formulation)
        bias_aug = self._intensity_transform_bias(
            bias_aug,
            low_range=low_range
        )
        # 3. gaussian smoothing
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        bias_aug = self._smooth_bias(
            bias_aug,
            sigma=sigma,
            kernel_size=kernel_size
        )
        # keep positive and bounded
        bias_aug = torch.clamp(bias_aug, min=clip_min, max=clip_max)
        return bias_aug

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.semi_warmup_iters=opt.semi_warmup_iters
        self.semi_supervised=opt.semi_supervised
        # define tensors
        self.input_V = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_U = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_F = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_W = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_M = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.TissueNorm_M = self.Tensor(opt.batchSize, opt.output_nc,opt.num_classes-1,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_shape=self.Tensor(3)

        self.input_W_peak=self.Tensor(1)
        self.i=0

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = True
            self.netTissueNorm = networks.define_TissueNorm(opt.imgSize,opt.num_classes,opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netTissueNorm, 'TN', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            opt.lrdd = opt.lr/2
            self.old_lrdd = opt.lrdd

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionSmooth = smoothing_loss()
            self.criterionSeg = SegLoss()
            self.criterionBMean = bias_mean_loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
            self.optimizer_TN = torch.optim.Adam(self.netTissueNorm.parameters(),
                                                lr=opt.lrdd, betas=(opt.beta1, 0.999),weight_decay=2e-5)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netTissueNorm)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_V = input['Ori' if AtoB else 'BF']
        input_B = input['BF' if AtoB else 'Ori']
        input_U = input['N4WI']
        input_F = input['FM']
        input_W = input['WM']
        input_W_peak=input['W_peak']
        input_M = input["M"]
        input_Shape=input["Image_shape"]
        TissueNorm_M=input["TissueNormM"]
        self.is_labeled=input["is_labeled"]
        self.input_V_path=input["img_paths"]

        self.input_V.resize_(input_V.size()).copy_(input_V)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_U.resize_(input_U.size()).copy_(input_U)
        self.input_F.resize_(input_F.size()).copy_(input_F)
        self.input_W.resize_(input_W.size()).copy_(input_W)
        self.input_M.resize_(input_M.size()).copy_(input_M)
        self.input_shape.resize_(input_Shape.size()).copy_(input_Shape)
        self.TissueNorm_M.resize_(TissueNorm_M.size()).copy_(TissueNorm_M)
        self.input_W_peak.resize_(input_W_peak.size()).copy_(input_W_peak)
        self.input_V = self.input_V
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def set_input_V(self, V):
        self.input_V = V

    def forward(self):
        self.real_V = Variable(self.input_V)
        self.fake_B = self.netG.forward(self.real_V)
        self.fake_B = torch.clip(self.fake_B,max=5,min=0.03)
        self.real_B = Variable(self.input_B)
        self.real_U = Variable(self.input_U)
        self.fake_U = self.real_V / self.fake_B
        self.fake_U = torch.clip(self.fake_U,max=15,min=0)

        if self.i%15==0:
            print(self.TissueNorm_M.shape)
            visShow(self.fake_U * self.input_W_peak, images,"U prediction")
            visShow(self.real_V * self.input_W_peak, images1, "source")
            visShow(self.real_U * self.input_W_peak, images2, "U truth")
            visShow(self.real_B , images3,"B")
            visShow(self.fake_B , images4, "B prediction")
            visShow(self.input_W, images5, "W")
            visShow(self.input_F, images6, "F")
            visShow(self.input_M, images7, "M")
            visShow(self.TissueNorm_M[:, 1:2], images8, "tissueNorm")

    def test(self):
        with torch.no_grad():
            self.real_V = self.input_V
            self.fake_B = self.netG.forward(self.real_V)
            self.real_B = self.input_B
            self.real_U = self.input_U
            self.fake_U = self.real_V / self.fake_B
            self.fake_U = torch.clip(self.fake_U, max=15, min=0)

    def test_no_real(self):
        self.real_V = self.input_V
        self.fake_B = self.netG.forward(self.real_V)
        self.fake_U = self.real_V / self.fake_B
        self.fake_U = torch.clip(self.fake_U, max=15, min=0)
        return self.fake_B, self.fake_U

    def get_image_paths(self):
        return self.input_V_path

    def backward_TissueNorm(self):
        device = self.input_V.device

        self.loss_Luniform_train = torch.tensor(0.0, device=device)
        self.loss_Luniform_mean = torch.tensor(0.0, device=device)
        self.loss_LTN_train = torch.tensor(0.0, device=device)
        self.loss_C = torch.tensor(0.0, device=device)

        detached_fake_U = self.fake_U.detach()

        # (B, C, D, H, W) -> for current code use batch=1 setup
        num_classes = self.TissueNorm_M.shape[1]
        print("tissueNorm_shape:",self.TissueNorm_M.shape)
        fake_U_repeated = detached_fake_U.repeat(num_classes, 1, 1, 1, 1)
        Uniform_U = torch.cat((self.TissueNorm_M.squeeze(0).unsqueeze(1), fake_U_repeated), dim=1)

        C_u = self.netTissueNorm.forward(Uniform_U)
        C_u = torch.clamp(C_u, 0.2, 8)

        uniform_template = torch.zeros_like(self.fake_U)
        for num in range(num_classes):
            tissue_mask = self.TissueNorm_M[:, num:num+1, :, :, :]
            tissue_pixels = detached_fake_U * tissue_mask
            normalized_tissue_pixels = tissue_pixels * C_u[num]
            uniform_template += normalized_tissue_pixels

        # combined_mask = (self.TissueNorm_M.sum(dim=1, keepdim=True) > 0).float()
        # masked_uniform_template = uniform_template * combined_mask
        #
        # if self.i % 15 == 0:
        #     visShow(masked_uniform_template, images9, "uniform_template")
        #
        # num_pixels = combined_mask.sum()
        # if num_pixels > 0:
        #     mean_unorm = masked_uniform_template.sum() / num_pixels
        #     std_unorm = torch.sqrt(
        #         (((masked_uniform_template - mean_unorm) ** 2) * combined_mask).sum() / num_pixels
        #     )
        #     self.loss_Luniform_train = std_unorm / (mean_unorm + 1e-8)
        #     self.loss_Luniform_mean = torch.abs(1 - mean_unorm)
        # else:
        #     self.loss_Luniform_train = torch.tensor(0.0, device=device)
        #     self.loss_Luniform_mean = torch.tensor(0.0, device=device)
        combined_mask = (self.TissueNorm_M.sum(dim=1, keepdim=True) > 0).float()
        masked_uniform_template = uniform_template * combined_mask
        if self.i % 15 == 0:
            visShow(masked_uniform_template, images9, "uniform_template")
        # -------------------------
        # 1) global uniformity loss
        # -------------------------
        num_pixels = combined_mask.sum()
        if num_pixels > 0:
            mean_unorm = masked_uniform_template.sum() / (num_pixels + 1e-8)
            var_unorm = (((masked_uniform_template - mean_unorm) ** 2) * combined_mask).sum() / (num_pixels + 1e-8)
            std_unorm = torch.sqrt(var_unorm + 1e-8)
            self.loss_Luniform_train = std_unorm / (mean_unorm + 1e-8)
        else:
            self.loss_Luniform_train = torch.tensor(0.0, device=device)
        # ---------------------------------------
        # 2) per-organ mean-to-1 normalization loss
        # ---------------------------------------
        self.loss_Luniform_mean = torch.tensor(0.0, device=device)
        valid_region_count = 0
        num_classes = self.TissueNorm_M.shape[1]
        for num in range(num_classes):
            tissue_mask = self.TissueNorm_M[:, num:num + 1, :, :, :]  # [B,1,D,H,W]
            tissue_pixels = uniform_template * tissue_mask
            tissue_num_pixels = tissue_mask.sum()
            if tissue_num_pixels > 0:
                tissue_mean = tissue_pixels.sum() / (tissue_num_pixels + 1e-8)
                self.loss_Luniform_mean += torch.abs(tissue_mean - 1.0)
                valid_region_count += 1
        if valid_region_count > 0:
            self.loss_Luniform_mean = self.loss_Luniform_mean / valid_region_count
        else:
            self.loss_Luniform_mean = torch.tensor(0.0, device=device)
        print("loss_Luniform_train:",self.loss_Luniform_train)
        print("loss_Luniform_mean:",self.loss_Luniform_mean)
        print("train_C_u", C_u)

        self.loss_LTN_train = self.loss_Luniform_train*2 + self.loss_Luniform_mean
        self.loss_LTN_train.backward(retain_graph=True)

        C_u_copy = C_u.detach()
        detached_fake_B = self.fake_B.detach()
        detached_fake_B_aug = self.augment_bias_field(
            detached_fake_B,
            max_rot_deg=5.0,
            max_shift=4,
            low_range=(1200, 8000),
            sigma_range=(0.5, 1.5),
            kernel_size=5,
            clip_min=0.03,
            clip_max=5.0
        )

        V_gen = detached_fake_B_aug * detached_fake_U

        if self.i % 15 == 0:
            visShow(V_gen, images11, "V_gen")

        real_V_repeated = V_gen.detach().repeat(num_classes, 1, 1, 1, 1)
        Uniform_V = torch.cat((self.TissueNorm_M.squeeze(0).unsqueeze(1), real_V_repeated), dim=1)

        C_v = self.netTissueNorm.forward(Uniform_V)
        C_v = torch.clamp(C_v, 0.2, 10)

        self.loss_C = self.criterionMSE(C_u_copy, C_v) * 1.0
        print("loss_C:",self.loss_C)
        self.loss_C.backward()

    def backward_G(self):
        self.loss_G=torch.tensor(0).to("cuda")
        # -----------------------------
        # 1) supervised loss for labeled
        # -----------------------------
        if self.is_labeled:
            self.loss_smooth = self.criterionSmooth(self.fake_B) * self.opt.lambda_E
            self.loss_G_S1 = self.criterionSeg(self.real_V, self.fake_B, self.input_W,
                                               self.input_W_peak) * self.opt.lambda_F
            self.loss_G_S2 = self.criterionSeg(self.real_V, self.fake_B, self.input_F,
                                               self.input_W_peak) * self.opt.lambda_F
            self.loss_B_mean = self.criterionBMean(self.fake_B, self.input_W, self.input_F)
            print("-----------------------------------------------------------------------------------")
            print("loss_smooth:",self.loss_smooth)
            print("loss_G_S1:",self.loss_G_S1)
            print("loss_G_S2:",self.loss_G_S2)
            print("self.loss_B_mean:",self.loss_B_mean)


            self.loss_G = self.loss_smooth+ self.loss_G_S1 + self.loss_G_S2+self.loss_B_mean
            print(self.loss_G)
            # labeled semi loss after warmup
            if self._is_in_semi_phase():
                print("TN")
                num_tissues = self.TissueNorm_M.shape[1]
                fake_U_repeated = self.fake_U.repeat(num_tissues, 1, 1, 1, 1)
                Uniform_U = torch.cat((self.TissueNorm_M.squeeze(0).unsqueeze(1), fake_U_repeated), dim=1)

                C_u = self.netTissueNorm.forward(Uniform_U)
                C_u = torch.clamp(C_u, 0.2, 8)
                print("infer_C_u",C_u)
                C_u_detach = C_u.detach()

                uniform_template = torch.zeros_like(self.fake_U)
                for num in range(num_tissues):
                    tissue_mask = self.TissueNorm_M[:, num:num+1, :, :, :]
                    tissue_pixels = self.fake_U * tissue_mask
                    normalized_tissue_pixels = tissue_pixels * C_u_detach[num]
                    uniform_template += normalized_tissue_pixels

                combined_mask = (self.TissueNorm_M.sum(dim=1, keepdim=True) > 0).float()
                masked_uniform_template = uniform_template * combined_mask

                if self.i % 15 == 0:
                    visShow(masked_uniform_template, images10, "uniform_template1")

                num_pixels = combined_mask.sum()
                if num_pixels > 0:
                    mean_unorm = masked_uniform_template.sum() / num_pixels
                    std_unorm = torch.sqrt(
                        (((masked_uniform_template - mean_unorm) ** 2) * combined_mask).sum() / num_pixels
                    )
                    self.loss_Luniform_infer = std_unorm / (mean_unorm + 1e-8) * self.opt.lambda_I
                    print("loss_Luniform_infer:",self.loss_Luniform_infer)

                    self.loss_G = self.loss_G + self.loss_Luniform_infer

        # -----------------------------
        # 3) consistency regularization after warmup
        # for both labeled and unlabeled
        # -----------------------------
        if self._is_in_semi_phase():
            print("CR")
            detached_fake_B = self.fake_B.detach().clamp(min=0.03, max=5.0)
            detached_fake_U = self.fake_U.detach().clamp(min=0.0, max=15.0)
            detached_fake_B_aug = self.augment_bias_field(
                detached_fake_B,
                max_rot_deg=5.0,
                max_shift=4,
                low_range=(1200, 8000),
                sigma_range=(0.5, 1.5),
                kernel_size=3,
                clip_min=0.03,
                clip_max=5.0
            ).detach()
            augment_image = detached_fake_U * detached_fake_B_aug
            augment_B = self.netG.forward(augment_image)
            augment_B = torch.clamp(augment_B, max=5, min=0.05)
            augment_U = augment_image / augment_B
            augment_U = torch.clamp(augment_U, min=0, max=15)
            self.loss_CR = self.criterionL1(augment_U, detached_fake_U)
            print("loss_CR:",self.loss_CR)
            self.loss_G = self.loss_G + self.loss_CR

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # 1. update TissueNorm only when labeled and after warmup
        if self.is_labeled:

            self.optimizer_TN.zero_grad()
            self.backward_TissueNorm()
            self.optimizer_TN.step()
            self.optimizer_TN.zero_grad()

        # 2. update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

        torch.cuda.memory_summary()

        self.i += 1


    def get_current_visuals(self):
        self.tissue_W = self.input_W.type(torch.ByteTensor)
        self.tissue_F = self.input_F.type(torch.ByteTensor)
        self.tissue_W = Variable(self.tissue_W)
        self.tissue_F = Variable(self.tissue_F)
        self.tissue_W = self.tissue_W.cuda()
        self.tissue_F = self.tissue_F.cuda()
        self.real_Vr = self.real_V * self.input_W_peak
        self.fake_Br = self.fake_B
        self.real_Ur = self.real_U * self.input_W_peak

        self.fake_Ur = self.fake_U * self.input_W_peak
        real_V = util.tensor2im3d(self.real_Vr.data)
        fake_B = util.tensor2im3d(self.fake_Br.data)
        real_B = util.tensor2im3d(self.real_B.data)
        fake_U = util.tensor2im3d(self.fake_Ur.data)

        real_U = util.tensor2im3d(self.real_Ur.data)
        return OrderedDict(
            [('real_V', real_V), ('real_U', real_U), ('fake_B', fake_B), ('real_B', real_B), ('fake_U', fake_U)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netTissueNorm, 'TN', label, self.gpu_ids)

    def update_learning_rate(self,epoch):
        lr = self.opt.lr *(self.opt.niter+self.opt.niter_decay-epoch)/self.opt.niter_decay
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
        lrddd = self.opt.lr/3 *(self.opt.niter+self.opt.niter_decay-epoch)/self.opt.niter_decay

        for param_group in self.optimizer_TN.param_groups:
            param_group['lr'] = lrddd
        print('update learning rate: %f -> %f' % (self.old_lrdd, lrddd))
        self.old_lrdd = lrddd
        print("g lr",self.old_lr)
        print("d lr",self.old_lrdd)

