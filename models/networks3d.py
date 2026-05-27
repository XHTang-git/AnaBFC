import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np



###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # Kaiming初始化
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if hasattr(m, 'bias') and m.bias is not None:
#             m.bias.data.fill_(0)
#     elif classname.find('BatchNorm3d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        print(1111111111111111111112321231313312323)
        netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        # print('G', gpu_ids)
        netG.cuda(device=gpu_ids[0])
        # netG.to(gpu_ids[0])
        # netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        # print('d',gpu_ids)
        netD.cuda(device=gpu_ids[0])
        # netD.to(gpu_ids[0])
        # netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD

def define_TissueNorm(img_size,num_classes,ndf,
             n_layers=5, norm='batch', use_sigmoid=False, gpu_ids=[]):
    net_TissueNorm = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    net_TissueNorm=TissueNormModel(img_size,num_classes,2, ndf, n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                               gpu_ids=gpu_ids)

    if use_gpu:
        net_TissueNorm.cuda(device=gpu_ids[0])
    net_TissueNorm.apply(weights_init)
    return net_TissueNorm


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        # if use_lsgan:
        #     self.loss = nn.MSELoss()
        # else:
        #     self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):

        target_tensor = self.get_target_tensor(input, target_is_real)

        return self.loss(input, target_tensor)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
            model += [nn.Softplus()]
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # print("use_dropout",use_dropout)
            # if use_dropout:
            #     model = down + [submodule] + up + [nn.Dropout(0.5)]
            # else:
            #     model = down + [submodule] + up
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # print ('x is ', x.shape)
            # print ('self.model(x) is ', self.model(x).shape)
            return torch.cat([self.model(x), x], 1)


# class UnetSkipConnectionBlock(nn.Module):
#     def __init__(self, outer_nc, inner_nc,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#         # self.connectConv=nn.Conv3d(outer_nc,outer_nc,kernel_size=3,stride=1,padding=1,bias=use_bias)
#         downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         # downconv1 = nn.Conv3d(inner_nc,inner_nc,kernel_size=3,stride=1,padding=1,bias=use_bias)
#         DownConv=[downconv]
#         downrelu = nn.LeakyReLU(0.2, False)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(False)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             # upconv1 = nn.Conv3d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1)
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1,bias=use_bias)
#             # upconv1=nn.Conv3d(outer_nc,outer_nc,kernel_size=3,stride=1,padding=1)
#             down = DownConv
#             up = [uprelu, upconv]
#             # up = [uprelu, upconv1, upconv]
#             model = down + [submodule] + up
#         elif innermost:
#             # upconv1 = nn.Conv3d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1)
#             upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             # upconv1 = nn.Conv3d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=1)
#             down = [downrelu]+DownConv
#             up = [uprelu, upconv, upnorm]
#             # up = [uprelu, upconv1,upconv, upnorm]
#
#             model = down + up
#         else:
#             # upconv1 = nn.Conv3d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1)
#
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             # upconv1 = nn.Conv3d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=1)
#             down = [downrelu]+DownConv+[downnorm]
#             up = [uprelu, upconv, upnorm]
#             # up = [uprelu, upconv1,upconv, upnorm]
#
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.3)]
#             else:
#                 model = down + [submodule] + up
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         if self.outermost:
#             return self.model(x)
#         else:
#             # x = self.connectConv(x)
#             x=torch.cat([self.model(x), x], 1)
#             # print ('x is ', x.shape)
#             # print ('self.model(x) is ', self.model(x).shape)
#             return x
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        print(n_layers)
        for n in range(1, n_layers):
            # print(11111111111111111111111111111111)
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        # if use_sigmoid:
        # sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = self.model(input)
        print(torch.mean(x))
        return x

# # Defines the PatchGAN discriminator with the specified arguments.
class TissueNormModel(nn.Module):
    def __init__(self,img_size,num_classes, input_nc, ndf=8, n_layers=5, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(TissueNormModel, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d


        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        print(n_layers)
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.encoder = nn.Sequential(*sequence)
        # 关键修改：全局池化，避免写死 Linear 输入维度
        self.pool = nn.AdaptiveAvgPool3d(1)
        # sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]
        # in_features=img_size[0]*img_size[1]*img_size[2]/2^(n_layers*3)*num_classes
        #这里需要修改
        hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(ndf * nf_mult*(num_classes-1), hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus()
        )

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        input: (B, C, D, H, W)
        return: (B,)
        """
        x = self.encoder(input)
        # print("encoder out:", x.shape)
        x = self.pool(x)
        # print("pooled:", x.shape)
        x = x.flatten()
        # print("flatten:", x.shape)
        x = self.mlp(x)
        return x
