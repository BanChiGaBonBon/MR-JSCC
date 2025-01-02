# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from options.train_options import TrainOptions
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import exp
from torchsummary import summary

# from torch import tensor as Tensor
opt = TrainOptions().parse()
Tensor = TypeVar('torch.tensor')
###############################################################################
# Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

class Flatten(nn.Module):
  def forward(self, x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>    

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())        
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_E(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, first_add_C=0):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        first_kernel (int) -- the kernel size of the first conv layer
        first_add_C  (int) -- additional channels for the feedback mode

    Returns a generator
    Our current implementation provides two types of generators:
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, first_add_C=first_add_C)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, activation_=activation)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_E_jscc(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, first_add_C=0):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        first_kernel (int) -- the kernel size of the first conv layer
        first_add_C  (int) -- additional channels for the feedback mode

    Returns a generator
    Our current implementation provides two types of generators:
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder_jscc(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, first_add_C=first_add_C)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G_jscc(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator_jscc(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, activation_=activation)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'none']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##############################################################################
# Encoder
##############################################################################
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, first_add_C=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """   
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        
        model = [nn.ReflectionPad2d((first_kernel-1)//2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]
        
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        self.model_down = nn.Sequential(*model)
        # self.mod1 = modulation(min(ngf * mult * 2, max_ngf))
        # self.mod2 = modulation(min(ngf * mult * 2, max_ngf))

        # self.multihead_attn1 = AttentionModule(embed_dim=64, seq_len=min(ngf * mult * 2, max_ngf))
        # self.multihead_attn2 = AttentionModule(embed_dim=64, seq_len=min(ngf * mult * 2, max_ngf))
        model= []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        # for i in range(n_blocks):  
        #     model += [ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        self.res1 = ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        # self.model_res = nn.Sequential(*model)
        # self.attn1 = AttentionModule(in_dim=min(ngf * mult,max_ngf)+first_add_C,out_dim=min(ngf * mult,max_ngf)+first_add_C)
        # self.attn2 = AttentionModule(in_dim=min(ngf * mult,max_ngf)+first_add_C,out_dim=min(ngf * mult,max_ngf)+first_add_C)
        if opt.is_ga:
            self.norm1 = nn.LayerNorm(min(ngf * mult * 2, max_ngf))
            self.norm2 = nn.LayerNorm(min(ngf * mult * 2, max_ngf))
            self.ga = GroupAttention(dim=min(ngf * mult,max_ngf)+first_add_C,num_heads=16,ws=2)
            self.mlp = Mlp(in_features=min(ngf * mult,max_ngf)+first_add_C, hidden_features=(min(ngf * mult,max_ngf)+first_add_C)*4, act_layer=nn.GELU, drop=0)

        


        self.projection = nn.Conv2d(min(ngf * mult,max_ngf)+first_add_C, C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)
        # self.projection = nn.Conv2d(min(ngf * mult,max_ngf)+first_add_C, C_channel, kernel_size=1, padding=0, stride=1, bias=use_bias)
        
    def forward(self, input, H=None,v=None):
        # print("input",input.shape)
        # v = v / opt.v_max
        z =  self.model_down(input)
        N,C,HH,WW = z.shape
        # z = self.mod1(self.res1(z),v)
        z = self.res1(z)

        z = self.res2(z)
        N,C,HH,WW = z.shape
        if opt.is_ga:
            z = z.flatten(2).permute(0,2,1)
            z = z + self.ga(self.norm1(z),HH,WW)
            z = z + self.mlp(self.norm2(z))
            z = z.permute(0,2,1).contiguous().view(N,C,HH,WW)
        # z = self.attn2(self.attn1(z))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        # print("z",z.shape)
        if H is not None:
            N,C,HH,WW = z.shape            
            z = torch.cat((z,H.contiguous().permute(0,1,2,4,3).view(N, -1, HH,WW)), 1)
        # return  self.projection(self.model_res(z))
        return self.projection(z)
        # return self.attn(z)


class Encoder_jscc(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, first_add_C=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """   
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder_jscc, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        
        model = [#nn.ReflectionPad2d((first_kernel-1)//2),
                #  nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(3),
                 activation]
        
        # add downsampling layers
        # for i in range(n_downsampling):
        #     mult = 2**i
        #     model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=5, stride=2, padding=1, bias=use_bias),
        #               norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        model += [nn.Conv2d(input_nc,16, kernel_size=5, stride=2, padding=2, bias=use_bias),
                      norm_layer(16), activation]
        model += [nn.Conv2d(16,32, kernel_size=5, stride=2, padding=2, bias=use_bias),
                      norm_layer(32), activation]

        self.model_down = nn.Sequential(*model)
        # self.mod1 = modulation(min(ngf * mult * 2, max_ngf))
        # self.mod2 = modulation(min(ngf * mult * 2, max_ngf))

        # self.multihead_attn1 = AttentionModule(embed_dim=64, seq_len=min(ngf * mult * 2, max_ngf))
        # self.multihead_attn2 = AttentionModule(embed_dim=64, seq_len=min(ngf * mult * 2, max_ngf))
        model= []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        # for i in range(n_blocks):  
        #     model += [ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        # self.res1 = ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        # self.res2 = ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        # self.model_res = nn.Sequential(*model)
        # self.attn1 = AttentionModule(in_dim=min(ngf * mult,max_ngf)+first_add_C,out_dim=min(ngf * mult,max_ngf)+first_add_C)
        # self.attn2 = AttentionModule(in_dim=min(ngf * mult,max_ngf)+first_add_C,out_dim=min(ngf * mult,max_ngf)+first_add_C)
        if opt.is_ga:
            self.norm1 = nn.LayerNorm(min(ngf * mult * 2, max_ngf))
            self.norm2 = nn.LayerNorm(min(ngf * mult * 2, max_ngf))
            self.ga = GroupAttention(dim=min(ngf * mult,max_ngf)+first_add_C,num_heads=16,ws=2)
            self.mlp = Mlp(in_features=min(ngf * mult,max_ngf)+first_add_C, hidden_features=(min(ngf * mult,max_ngf)+first_add_C)*4, act_layer=nn.GELU, drop=0)
        model2= []
        model2 += [nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=use_bias),
                      norm_layer(32), activation]
        model2 += [nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=use_bias),
                      norm_layer(32), activation]

        self.conv2 = nn.Sequential(*model2)
        


        self.projection = nn.Conv2d(min(ngf * mult,max_ngf)+first_add_C, C_channel, kernel_size=5, padding=2, stride=1, bias=use_bias)
        # self.projection = nn.Conv2d(min(ngf * mult,max_ngf)+first_add_C, C_channel, kernel_size=1, padding=0, stride=1, bias=use_bias)
        
    def forward(self, input, H=None,v=None):
        # print("input",input.shape)
        # v = v / opt.v_max
        z =  self.model_down(input)
        N,C,HH,WW = z.shape
        # z = self.mod1(self.res1(z),v)
        # z = self.res1(z)

        # z = self.res2(z)
        z = self.conv2(z)

        N,C,HH,WW = z.shape
        if opt.is_ga:
            z = z.flatten(2).permute(0,2,1)
            z = z + self.ga(self.norm1(z),HH,WW)
            z = z + self.mlp(self.norm2(z))
            z = z.permute(0,2,1).contiguous().view(N,C,HH,WW)
        # z = self.attn2(self.attn1(z))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        # print("z",z.shape)
        if H is not None:
            N,C,HH,WW = z.shape            
            z = torch.cat((z,H.contiguous().permute(0,1,2,4,3).view(N, -1, HH,WW)), 1)
        # return  self.projection(self.model_res(z))
        return self.projection(z)
    
##############################################################################
# Generator
##############################################################################
class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Generator, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        # model1 = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]
        # self.model1 = nn.Sequential(*model1)
        # model = []
        # modelt = [TransformerModel(ngf_dim,64, 4, 2, 128, opt.S)]
        # self.modelt = nn.Sequential(*modelt)



        # model_up = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]
        # for i in range(n_blocks):
        #     model_up += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        # self.model_up = nn.Sequential(*model_up)


        self.conv1 = nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)
        if opt.is_ga:
            self.norm1 = nn.LayerNorm(ngf_dim)
            self.norm2 = nn.LayerNorm(ngf_dim)
            self.ga = GroupAttention(dim=ngf_dim,num_heads=16,ws=2)
            self.mlp = Mlp(in_features=ngf_dim, hidden_features=ngf_dim*4, act_layer=nn.GELU, drop=0)
        # self.conv1 = nn.Conv2d(C_channel,ngf_dim,kernel_size=1, padding=0 ,stride=1, bias=use_bias)
        # self.attn1 = AttentionModule(in_dim=ngf_dim,out_dim=ngf_dim)
        # self.attn2 = AttentionModule(in_dim=ngf_dim,out_dim=ngf_dim)
        self.res1 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        # self.multihead_attn1 = AttentionModule(embed_dim=64, seq_len=ngf_dim)
        # self.multihead_attn2 = AttentionModule(embed_dim=64, seq_len=ngf_dim)
        # self.mod1 = modulation(ngf_dim)
        # self.mod2 = modulation(ngf_dim)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult //2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((first_kernel-1)//2), nn.Conv2d(ngf, output_nc, kernel_size=first_kernel, padding=0)]

        if activation_ == 'tanh':
            model +=[nn.Tanh()]
        elif activation_ == 'sigmoid':
            model +=[nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input,v=None):
        
        # y1 = self.model1(input)
        # y2 = y1.permute(0,2,3,1).view(input.shape[0],opt.S,-1)
        # input = self.modelt(y2).permute(0,2,1).view(y1.size())

        # z = self.mod(self.model_up(input),v)
        # v = v / opt.v_max
        # z = self.res1(self.mod1(self.conv1(input),v))
        # z = self.res1(self.conv1(input))

        z = self.conv1(input)
        N,C,HH,WW = z.shape
        if opt.is_ga:
            z = z.flatten(2).permute(0,2,1)
            z = z + self.ga(self.norm1(z),HH,WW)
            z = z + self.mlp(self.norm2(z))
            z = z.permute(0,2,1).contiguous().view(N,C,HH,WW)
        # z = self.attn(z)
        
        # z = self.multihead_attn1(v,z.view(N,C,-1))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        # z = self.res2(self.mod2(z,v))
        z = self.res1(z)
        z = self.res2(z)
        # z = self.attn2(self.attn1(z))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        if self.activation_=='tanh':
            return self.model(z)
        elif self.activation_=='sigmoid':
            return 2*self.model(z)-1
        
class Generator_jscc(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Generator_jscc, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        # model1 = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]
        # self.model1 = nn.Sequential(*model1)
        # model = []
        # modelt = [TransformerModel(ngf_dim,64, 4, 2, 128, opt.S)]
        # self.modelt = nn.Sequential(*modelt)



        # model_up = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]
        # for i in range(n_blocks):
        #     model_up += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        # self.model_up = nn.Sequential(*model_up)


        self.conv1 = nn.ConvTranspose2d(C_channel,ngf_dim,kernel_size=5, padding=2 ,stride=1, bias=use_bias)
        if opt.is_ga:
            self.norm1 = nn.LayerNorm(ngf_dim)
            self.norm2 = nn.LayerNorm(ngf_dim)
            self.ga = GroupAttention(dim=ngf_dim,num_heads=16,ws=2)
            self.mlp = Mlp(in_features=ngf_dim, hidden_features=ngf_dim*4, act_layer=nn.GELU, drop=0)
        # self.conv1 = nn.Conv2d(C_channel,ngf_dim,kernel_size=1, padding=0 ,stride=1, bias=use_bias)
        # self.attn1 = AttentionModule(in_dim=ngf_dim,out_dim=ngf_dim)
        # self.attn2 = AttentionModule(in_dim=ngf_dim,out_dim=ngf_dim)
        # self.res1 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        # self.res2 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        model= []
        model += [nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2, bias=use_bias),
                      norm_layer(32), activation]
        model += [nn.ConvTranspose2d(32,32, kernel_size=5, stride=1, padding=2, bias=use_bias),
                      norm_layer(32), activation]
        self.conv2 = nn.Sequential(*model)
        # self.multihead_attn1 = AttentionModule(embed_dim=64, seq_len=ngf_dim)
        # self.multihead_attn2 = AttentionModule(embed_dim=64, seq_len=ngf_dim)
        # self.mod1 = modulation(ngf_dim)
        # self.mod2 = modulation(ngf_dim)
        model = []
        # for i in range(n_downsampling):
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(min(ngf * mult //2, max_ngf)),
        #               activation]
        model += [nn.ConvTranspose2d(32, 16,
                                         kernel_size=5, stride=2,
                                         padding=2, output_padding=1,
                                         bias=use_bias),
                      norm_layer(16),
                      activation]
        
        model += [nn.ConvTranspose2d(16, output_nc, stride=2,kernel_size=first_kernel, padding=2,output_padding=1),norm_layer(3)]

        if activation_ == 'tanh':
            model +=[nn.Tanh()]
        elif activation_ == 'sigmoid':
            model +=[nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input,v=None):
        
        # y1 = self.model1(input)
        # y2 = y1.permute(0,2,3,1).view(input.shape[0],opt.S,-1)
        # input = self.modelt(y2).permute(0,2,1).view(y1.size())

        # z = self.mod(self.model_up(input),v)
        # v = v / opt.v_max
        # z = self.res1(self.mod1(self.conv1(input),v))
        # z = self.res1(self.conv1(input))

        z = self.conv1(input)
        N,C,HH,WW = z.shape
        if opt.is_ga:
            z = z.flatten(2).permute(0,2,1)
            z = z + self.ga(self.norm1(z),HH,WW)
            z = z + self.mlp(self.norm2(z))
            z = z.permute(0,2,1).contiguous().view(N,C,HH,WW)
        # z = self.attn(z)
        z = self.conv2(z)
        # z = self.multihead_attn1(v,z.view(N,C,-1))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        # z = self.res2(self.mod2(z,v))
        # z = self.res1(z)
        # z = self.res2(z)
        # z = self.attn2(self.attn1(z))
        # z = self.multihead_attn2(v,z.view(N,C,-1)).view(N,C,HH,WW)
        if self.activation_=='tanh':
            return self.model(z)
        elif self.activation_=='sigmoid':
            return 2*self.model(z)-1
        
        

#########################################################################################
# Residual block
#########################################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map
        
        
        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers+1):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model'+str(self.n_layers+1))
        out = model(res[-1])

        return res[1:], out


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


#########################################################################################
# Residual-like subnet
#########################################################################################
class Subnet(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(Subnet, self).__init__()
        self.conv_block = self.build_conv_block(dim, dim_out, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, 64, kernel_size=3, padding=p, bias=use_bias), norm_layer(64), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(64, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_block(x)  # add skip connections
        # print(out.shape)
        return out


def define_Subnet(dim, dim_out, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    net = Subnet(dim=dim, dim_out=dim_out, padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)    
    return init_net(net, init_type, init_gain, gpu_ids)


class OFDMRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        初始化RNN模型。
        :param input_size: 输入数据的特征数，这里是M（子载波数量）。
        :param hidden_size: RNN隐藏层的大小。
        :param num_layers: RNN的层数。
        :param output_size: 输出层的大小，通常与输入大小相同，除非有特定需求。
        """
        super(OFDMRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,dropout = 0.5)
        
        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """                                                              
        前向传播定义。
        :param x: 输入数据，维度为 (N, S, M)。
        :return: RNN的输出。
        """
   
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)

        # 变换输出维度以适应全连接层
        out = out.reshape(out.shape[0], -1, self.hidden_size)

        # 应用全连接层
        out = self.fc(out)

        # 输出维度为(N, S, M)，与输入维度一致
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)

        # self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, input_size))
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, input_size)
        # self.decoder = Mlp(in_features=d_model, out_features=input_size)
        
    def forward(self, src):
        src = self.input_linear(src)
        # src = src.clone
        src += self.positional_encoding[:, :src.size(1)]

        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output
    
class modulation(nn.Module):
    def __init__(self, C_channel):

        super(modulation, self).__init__()

        activation = nn.ReLU(True)
        # Policy network
        model_multi = [nn.Linear(C_channel + 1, C_channel), activation,
                       nn.Linear(C_channel, C_channel), nn.Sigmoid()]

        model_add = [nn.Linear(C_channel + 1, C_channel), activation,
                     nn.Linear(C_channel, C_channel)]

        self.model_multi = nn.Sequential(*model_multi)
        self.model_add = nn.Sequential(*model_add)

    def forward(self, z, V):

        # Policy/gate network
        N, C, W, H = z.shape
        V = V/opt.v_max
        z_mean = torch.mean(z, (-2, -1))
        z_cat = torch.cat((z_mean, V), -1)
       
        factor = self.model_multi(z_cat).view(N, C, 1, 1)
        addition = self.model_add(z_cat).view(N, C, 1, 1)

        return z * factor + addition


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
        #@save
    def transpose_qkv(X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
        # num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])


    #@save
    def transpose_output(X, num_heads):
        """逆转transpose_qkv函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

class AttentionModule(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(AttentionModule, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.out_dim, width, height)
        out = self.gamma * out + x
        
        
        return out
    
class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=3):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads 
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.pe = nn.Parameter(torch.randn(1,64,256))
        self.pe = nn.Parameter(torch.randn(1,1024,256))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        x += self.pe
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x