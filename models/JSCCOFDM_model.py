# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import random
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
import models.networks as networks
import models.channel as channel
import matplotlib.pyplot as plt
from models.utils import normalize, ZF_equalization, MMSE_equalization, LS_channel_est, LMMSE_channel_est#,orth_dist,deconv_orth_dist
def seed_everything(seed=526):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getG(M, N, chanEst, padLen, padType,device):
    """
    Form time domain channel matrix from detected DD paths.
    Parameters:
        M (int): FFT size
        N (int): Number of subsymbols
        chanEst (dict): Contains 'pathGains', 'pathDelays', 'pathDopplers'
        padLen (int): Padding length for ZP or CP
        padType (str): Padding type ('ZP', 'CP', or 'None')
    Returns:
        G (torch.Tensor): Channel matrix of size (batch_size, MN, MN)
    """
    batch_size = chanEst['pathGains'].size(0)

    Meff = M #+ padLen  # account for subsymbol pad length in forming channel
    lmax = padLen      # max delay

    
    MN = Meff * N
    # P = chanEst['pathGains'].size(1)  # number of paths
    # print(lmax)
    # Initialize g array
    g = torch.zeros(batch_size, lmax+1, MN).to(torch.complex128).to(device)


    # Compute the channel response for each path
    pathGains = chanEst['pathGains']
    # print(1)
    pathDelays = chanEst['pathDelays']

    pathDopplers = chanEst['pathDopplers']
    pathDelays = torch.clamp(pathDelays, 0, padLen-1)
    # print(pathGains.shape)
    # print(pathDelays[0,:])
    # print(pathDopplers[0,:])
    # print(pathGains[0,:])
    # print(indices.shape)
    
    # indices = torch.arange(MN).to(device).unsqueeze(0).unsqueeze(0) - pathDelays.unsqueeze(-1)
    # print(indices.shape)
    # print(pathDelays.shape)
    # exit()
    # print(indices)
    # phase_shift = torch.exp(1j * 2 * torch.pi / MN * pathDopplers.unsqueeze(-1) * indices)

    # Scatter the contributions to the g matrix
    # print(g.dtype)
    # print(pathDelays.dtype)
    # print(pathGains.dtype)
    # print(phase_shift.dtype)
    # print(torch.max(pathDelays))
    # g = g.scatter_add(1, pathDelays.unsqueeze(-1).expand(-1, -1, MN), (pathGains.unsqueeze(-1) * phase_shift))
    # g.permute(1,0,2)
    g = g.reshape(-1,MN)
    for p in range(0,padLen):
        ind = torch.arange(MN).unsqueeze(0).repeat(batch_size,1).to(device)
        
        ind = ind-pathDopplers[:,p].unsqueeze(-1)
        # print(ind)
        ps =  pathGains[:,p].unsqueeze(-1)*torch.exp(1j*2*torch.pi/MN*pathDopplers[:,p].unsqueeze(-1)*ind)
        # print(ps.shape)
        # print(pathDelays[:,p]+1)
        
        rows, cols = pathDelays.shape

        # 创建一个行向量，其中每个元素为行号乘以列数
        row_offsets = torch.tensor(np.arange(rows)[:,np.newaxis]*cols).to(device)
        # print(row_offsets.shape)
        
        # print(row_offsets[:,0])
      
        
        # print(row_offsets.reshape(-1,1))
        # print(row_offsets)
        # print(pathDelays.shape)
        # 将行偏移量添加到原始张量
        modified_tensor_2d = pathDelays + row_offsets

        # 将二维张量展平成一维张量
        pd1 = modified_tensor_2d.t().flatten()
        # print(pd1)
        # print(pd1[batch_size*p:batch_size*(p+1)])
        g[pd1[batch_size*p:batch_size*(p+1)],:] =   g[pd1[batch_size*p:batch_size*(p+1)],:] + ps 
        # g[:,pathDelays[:,p]+1,:] = g[:,pathDelays[:,p]+1,:] +ps.unsqueeze(1)
    g = g.reshape(batch_size,-1,MN)
    # Form the MN-by-MN channel matrix G
    G = torch.zeros(batch_size, MN, MN, dtype=torch.cfloat).to(chanEst['pathGains'].device)

    maxd = 5
    for l in range(0,maxd+1):
        # mask = pathDelays == l
        diag_g = torch.diag_embed(g[:, l, l:],offset = -l)
        # print(diag_g.shape)
        # G += torch.cat([torch.zeros(batch_size, MN, l.item(), dtype=torch.cfloat).to(G.device), diag_g], dim=-1)
        G += diag_g

    return G

def conv_orth_dist(kernel, stride = 1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h),"Do not support rectangular kernel"
    #half = np.floor(w/2)
    assert stride<w,"Please use matrix orthgonality instead"
    new_s = stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s*new_s*i_c, -1))
    Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
    temp= np.zeros((i_c, i_c*new_s**2))
    for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
    return torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )
    
def deconv_orth_dist(kernel, stride = 2, padding = 1):

    [o_c, i_c, w, h] = kernel.shape
    
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )
    
def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())

class JSCCOFDMModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'PAPR', 'CE', 'EQ','valid','OR']
        self.loss_valid = 0.0
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none':
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']
        if self.opt.is_t:
            self.model_names = ['E', 'G','T']

        if self.opt.feedforward in ['EXPLICIT-RES']:
            self.model_names += ['S1', 'S2']
        
        if self.opt.feedforward in ['EXPLICIT-CE-EQ', 'EXPLICIT-RES','EXPLICIT-EQ','JSCC']:
            C_decode = opt.C_channel
        elif self.opt.feedforward == 'IMPLICIT':
            # C_decode = opt.C_channel + self.opt.N_pilot*self.opt.P*2 + self.opt.P*2 
            C_decode = opt.C_channel
        elif self.opt.feedforward == 'EXPLICIT-CE':
            C_decode = opt.C_channel + self.opt.P * 2

        else:
            print("wrong feedforward")
        if self.opt.is_feedback:
            add_C = self.opt.P*2
        else:
            add_C = 0
        if self.opt.feedforward in ['JSCC']:
            self.netE = networks.define_E_jscc(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=32,
                                        n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                        n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=5, first_add_C=add_C)

            self.netG = networks.define_G_jscc(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=32,
                                        n_downsample=opt.n_downsample, C_channel=C_decode, 
                                        n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=5, activation=opt.activation)
        else:
            # define networks (both generator and discriminator)
            self.netE = networks.define_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                        n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                        n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, first_add_C=add_C)

            self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                        n_downsample=opt.n_downsample, C_channel=C_decode, 
                                        n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)
        

        #if self.isTrain and self.is_GAN:  # define a discriminator; 
        if self.opt.gan_mode != 'none':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, 
                                          opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.opt.feedforward in ['EXPLICIT-RES']:
            self.netS1 = networks.define_Subnet(dim=(self.opt.N_pilot*self.opt.P+1)*2, dim_out=self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            self.netS2 = networks.define_Subnet(dim=(self.opt.M+1)*self.opt.P*2, dim_out=self.opt.M*self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.opt.feedforward in ['EXPLICIT-CE']:
            self.netS1 = networks.define_Subnet(dim=(self.opt.N_pilot*self.opt.P+1)*2, dim_out=self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.opt.feedforward in ['EXPLICIT-EQ']:
            # self.netR = networks.init_net(networks.OFDMRNN((self.opt.M*self.opt.P+self.opt.N_pilot+1)*2,128,2,(self.opt.M*self.opt.P+self.opt.N_pilot+1)*2),gpu_ids=self.gpu_ids,init_type=opt.init_type)
            # self.netR.load_state_dict(torch.load('rnn_model_params.pth'))

            # input_size = (opt.M*opt.P+opt.N_pilot+1)*2  # 假设您的数据特征大小为M
            # d_model = 64  # Transformer内部的特征维度
            # nhead = 4  # 注意力机制中的头数
            # num_encoder_layers = 2  # 编码器层数
            # dim_feedforward = 128  # 前馈网络的维度
            # max_seq_length = opt.S  # 假设您的序列长度为S
 
            input_size = opt.M*2  # 假设您的数据特征大小为M
            d_model = self.opt.d_model  # Transformer内部的特征维度
            nhead = self.opt.nhead  # 注意力机制中的头数
            num_encoder_layers = 2  # 编码器层数
            dim_feedforward = 128  # 前馈网络的维度
            
            max_seq_length = opt.S*3  # 假设您的序列长度为S
            if self.opt.is_t :
                # self.netT = networks.init_net(networks.TransformerModel(input_size,opt.M*4, nhead, num_encoder_layers, dim_feedforward, max_seq_length),gpu_ids=opt.gpu_ids,init_type=opt.init_type)
                
                self.netT = networks.init_net(networks.TransformerModel(input_size,d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length),gpu_ids=opt.gpu_ids,init_type=opt.init_type)
                if self.opt.is_pb:
                    self.netTm = networks.init_net(networks.TransformerModel(opt.S*2,opt.S*2*3, nhead, num_encoder_layers, dim_feedforward,  opt.M*3),gpu_ids=opt.gpu_ids,init_type=opt.init_type)
                if self.isTrain:

                    # self.netT.load_state_dict(torch.load('trans_model_params7_50_6_10_6_256.pth'))
                    if self.opt.is_pb:
                        self.netTm.load_state_dict(torch.load('transm_model_params8_32_256.pth'))
           
        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())

            if self.opt.feedforward in ['EXPLICIT-RES']:
                params+=list(self.netS1.parameters()) + list(self.netS2.parameters())
            elif self.opt.is_t:
                params+=list(self.netT.parameters())
                if self.opt.is_pb:
                    params+=list(self.netTm.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        self.opt = opt
        self.ofdm = channel.OFDM(opt, self.device, self.opt.pilot_path)

    def name(self):
        return 'JSCCOFDM_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)
        
    def forward(self):
        
        N = self.real_A.shape[0]
        
        # if self.opt.is_random_v:
        #     # velocity = random.uniform(self.opt.v_min,self.opt.v_range)
        #     velocity = torch.rand(N, 1).to(self.device) * (self.opt.v_range-self.opt.v_min) + self.opt.v_min
        # else:
        #     # velocity = float(self.opt.V)
        #     # velocity = torch.ones(N, 1).to(self.device) * self.opt.V
        #     velocity = torch.rand(N, 1).to(self.device) * 20 + self.opt.V
            

        ##test
        velocity = torch.tensor(self.opt.V).view(1,1).repeat(int(N/self.opt.I),1).to(self.device)

        if self.opt.is_feedback:
            with torch.no_grad():
                cof, _ = self.ofdm.channel.sample(N, self.opt.P, self.opt.M, self.opt.L)
                out_pilot, H_t, noise_pwr = self.ofdm(None, SNR=self.opt.SNR, cof=cof, batch_size=N)
                H_est = self.channel_estimation(out_pilot, noise_pwr)
            H = torch.view_as_real(H_est).to(self.device)               
            latent = self.netE(self.real_A, H)
        else:
            cof = None
            latent = self.netE(self.real_A,v=velocity)
        # print(latent.shape)
        if self.opt.is_cm:
        # s32 m 12 s 64 m 6
            self.tx = latent.contiguous().view(N, self.opt.P, self.opt.M, 2, self.opt.S).contiguous().permute(0,1,4,2,3)
        # S 6 M 64 s 12 m 32
        else :
            self.tx = latent.contiguous().view(N, self.opt.P, self.opt.S, 2, self.opt.M).contiguous().permute(0,1,2,4,3)
        
        #s 12 m 32
        # self.tx = latent.contiguous().view(N, self.opt.P, self.opt.S, self.opt.M,2)


        #16 * 6 * 64
        # N = int(self.opt.batch_size/self.opt.I)
        # self.tx = latent.contiguous().view(N,self.opt.P,int(self.opt.S/self.opt.I), 2, self.opt.M).contiguous().permute(0,1,2,4,3)


        self.tx_c = torch.view_as_complex(self.tx.contiguous())
        N = int(self.opt.batch_size/self.opt.I)
        self.tx_c = normalize(self.tx_c, 1).view(N,self.opt.P,self.opt.S,self.opt.M)
        # self.tx_c = self.tx_c.view(N,self.opt.P,self.opt.S,self.opt.M)
        # N, C, H, W = latent.shape
        # print(torch.equal(torch.view_as_real(self.tx_c).view(N,1,self.opt.S,-1).view(int(self.opt.batch_size/self.opt.I),1,self.opt.S,self.opt.M,2).permute(0,1,2,4,3).contiguous().view(N, C, H, W),latent))
        
        if self.opt.is_pb:
                out_pilot_m,out_pilot_s, out_sig, self.H_true, noise_pwr, self.PAPR, self.PAPR_cp = self.ofdm(self.tx_c, SNR=self.opt.SNR, cof=None,velocity=velocity)
                N, C, H, W = latent.shape
                r3 = torch.view_as_real(out_sig).contiguous().view(N,1,self.opt.S,-1)
                r2 = torch.view_as_real(out_pilot_s).contiguous().view(N,1,self.opt.S,-1)
                r2 = torch.cat((r2[:,:,:,:self.opt.N_pilot].repeat(1,1,1,int(self.opt.M/self.opt.N_pilot)),r2[:,:,:,self.opt.N_pilot:].repeat(1,1,1,int(self.opt.M/self.opt.N_pilot))),-1)
                r1 = torch.view_as_real(self.ofdm.pilot[:self.opt.S]).view(1,1,self.opt.S,2).repeat(N,1,1,1)
                r1 = torch.cat((r1[:,:,:,:1].repeat(1,1,1,self.opt.M),r1[:,:,:,1:].repeat(1,1,1,self.opt.M)),-1)


                rnn_input = torch.cat((r3, r1, r2), 2).contiguous().view(N, -1,self.opt.M*2).float()
                # outputs = model(rnn_input)
                r3 = self.netT(rnn_input)[:,:self.opt.S,:].unsqueeze(1).contiguous().view(N,1,self.opt.S,self.opt.M,2).permute(0,1,2,4,3).contiguous().view(N,1,self.opt.S*2,self.opt.M)
                # r3 = torch.view_as_real(out_sig).contiguous().view(N,1,self.opt.S,self.opt.M,2).permute(0,1,2,4,3).reshape(N,1,self.opt.S*2,self.opt.M)
                r2 = torch.view_as_real(out_pilot_m).contiguous().view(N,1,self.opt.N_pilot,self.opt.M,2).permute(0,1,2,4,3).contiguous().view(N,1,-1,self.opt.M)
                r2 = torch.cat((r2[:,:,:self.opt.N_pilot,:].repeat(1,1,int(self.opt.S/self.opt.N_pilot),1),r2[:,:,self.opt.N_pilot:,:].repeat(1,1,int(self.opt.S/self.opt.N_pilot),1)),-2)
                r1 = torch.view_as_real(self.ofdm.pilot[:self.opt.M]).view(1,1,self.opt.M,2).permute(0,1,3,2)
                r1 = torch.cat((r1[:,:,:1,:].repeat(N,1,int(self.opt.S),1),r1[:,:,1:,:].repeat(N,1,int(self.opt.S),1)),-2)
                # print(r3.shape)
                # print(r2.shape)
                # print(r1.shape)
                rnn_input = torch.cat((r3, r1, r2), 3).contiguous().view(N, self.opt.S*2,self.opt.M*3).float()
                # print(rnn_input[1,:,:opt.M])
                # print(rnn_input[1,:,opt.M:opt.M*2])
                # print(rnn_input[1,:,opt.M*2:opt.M*3])
                # outputs = model2(rnn_input.permute(0,2,1)).permute(0,2,1)
                outputs = self.netTm(rnn_input.permute(0,2,1)).permute(0,2,1)
                # print(outputs.shape)
                
                self.rnn_output = outputs[:,:,:self.opt.M].contiguous().view(N,self.opt.S,2,self.opt.M).permute(0,1,3,2).contiguous().view(N,self.opt.S,self.opt.M*2)
                dec_in = self.rnn_output.reshape(N,self.opt.P,self.opt.S,self.opt.M,2).permute(0,1,3,4,2).reshape(N,C,H,W)
                # print(torch.equal(outputs,torch.view_as_real(out_sig).contiguous().view(N,opt.S,opt.M*2).float()))
                self.rnn_ori = torch.view_as_real(self.tx_c).contiguous().view(N,self.opt.S,-1).float()
                self.fake = self.netG(dec_in,v=velocity)

        else:
            out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, self.PAPR_cp = self.ofdm(self.tx_c, SNR=self.opt.SNR, cof=cof,velocity=velocity)
            
            
            # self.H_true = self.H_true.to(self.device)
            if self.opt.feedforward in ['EXPLICIT-EQ','JSCC'] :


                
                # r1 = torch.view_as_real(self.ofdm.pilot).view(1,1,self.opt.S,1,2).repeat(N,1,1,self.opt.M,1).contiguous().view(N,1,self.opt.S,-1)
                
                if self.opt.is_pm:
                    r1 = torch.view_as_real(self.ofdm.pilot).view(1,1,1,self.opt.M*2).repeat(N,1,2,1)
                else:
                    r1 = torch.view_as_real(self.ofdm.pilot).view(1,1,self.opt.S,2).repeat(N,1,1,1)
                    r1 = torch.cat((r1[:,:,:,:1].repeat(1,1,1,self.opt.M),r1[:,:,:,1:].repeat(1,1,1,self.opt.M)),-1)

                self.rnn_ori = torch.cat((torch.view_as_real(self.tx_c).contiguous().view(N,1,self.opt.S,-1),r1,r1),2).contiguous().view(N, -1,self.opt.M*2)
            N, C, H, W = latent.shape
            # print(torch.equal(self.tx.permute(0,1,2,4,3).contiguous().view(N, -1, H, W),latent))
            N = int(N/self.opt.I)
            # print("latent",latent.shape)
            if self.opt.feedforward == 'IMPLICIT':
                
                # r2 = torch.view_as_real(out_pilot).contiguous().view(N, -1, H, W)
                # r3 = torch.view_as_real(out_sig).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
                # p1 = self.ofdm.pilot.view(1,1,self.opt.S,1).repeat(N,1,1,1)
                # p2 = out_pilot[:,:,:,:1]
                # c = p1/ p2

                # dec_in = torch.view_as_real(c*out_sig)
                
                if(self.opt.mod=='OFDM'):
                    # r1 = torch.view_as_real(self.ofdm.pilot).repeat(N,1,1,1,1).contiguous().view(N, -1, H, W)
                    # dec_in = torch.cat((r3,r1, r2), 1).contiguous().view(N, -1, H, W).float()

                    # S 6 M 64
                    dec_in = torch.view_as_real(out_sig).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W).float()
                if(self.opt.mod=='OTFS'):
                    # r1 = torch.view_as_real(self.ofdm.pilot).view(1,1,self.opt.S,1,2).repeat(N,1,1,1,1)
                    dec_in = torch.cat((r1, r2, r3), 3)

                    # print("r1",r1.shape)
                    # print("r2",r2.shape)
                    # print("r3",r3.shape)
                    
                    # print("decin",dec_in.shape)

                    dec_in = dec_in.contiguous().permute(0,1,3,4,2).contiguous().view(N, -1, H, W).float()
                    # print(self.criterionL2( dec_in,latent))
                    # print("decin",dec_in.shape)
                    
                self.fake = self.netG(dec_in)
            elif self.opt.feedforward == 'EXPLICIT-CE':
                # Channel estimation
                # self.H_est = self.channel_estimation(out_pilot, noise_pwr)
            
                
                
                
                if self.opt.real_cof:
                    sub1_output =   torch.view_as_real(torch.sum(self.ofdm.channel.cof_fin,dim=2)).contiguous().view(N, -1, H, W).float()
                else:
                    sub11 = torch.view_as_real(self.ofdm.pilot).view(1,1,self.opt.S,1,2).repeat(N,1,1,1,1)
                    sub12 = torch.view_as_real(out_pilot)
                    sub1_input = torch.cat((sub11, sub12), 3).contiguous().permute(0,1,3,4,2).contiguous().view(N, -1, H, W).float()
                    sub1_output = self.netS1(sub1_input)
            
                r2 = torch.view_as_real(out_sig).contiguous().permute(0,1,3,4,2).contiguous().view(N, -1, H, W).float()             
                dec_in = torch.cat((sub1_output, r2), 1)
                self.fake = self.netG(dec_in)
            elif self.opt.feedforward == 'EXPLICIT-CE-EQ':
                if self.opt.modulation=='OFDM':
                        
                    self.H_est = self.channel_estimation(out_pilot, noise_pwr)
                    self.H_est = self.H_est.permute(0,1,3,2)
                    self.rx = self.equalization(self.H_est, out_sig, noise_pwr)
                    
                elif self.opt.modulation == 'OTFS':
                    # print((torch.abs(out_pilot[0,:,:])**2).sum())
                    # Hdd = out_pilot.squeeze() * ((4-4j)) / (4*np.sqrt(2)+noise_pwr)
                    pilot = torch.fft.fft2(self.ofdm.pilot_cp[0,0,:,:])
                    # print(pilot.shape)
                    Hdd = torch.fft.fft2(out_pilot.squeeze()) * pilot.conj() / (4*np.sqrt(2)+noise_pwr)
                    # Identify significant paths
                    # abs_Hdd = torch.abs(Hdd)
                    # top_k = 10
                    # topk_vals, topk_indices = torch.topk(abs_Hdd.reshape(Hdd.size(0), -1), top_k, dim=-1)

                    # # Convert flat indices to 3D indices
                    # batch_indices = torch.arange(Hdd.size(0)).unsqueeze(1).expand(-1, top_k).flatten()
                    
                    # topk_indices = topk_indices.flatten()
                    # # print(Hdd.shape)
                    # vp = (topk_indices // Hdd.size(2)).view(Hdd.size(0), top_k)
                    # lp = (topk_indices % Hdd.size(2)).view(Hdd.size(0), top_k)
                    # pathGains = Hdd[batch_indices,  vp.flatten(),lp.flatten()].reshape(Hdd.shape[0],-1)
                    # x = np.linspace(0,31,32)
                    # y = np.linspace(0,11,12)
                    # x,y = np.meshgrid(y,x)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    
                    # # 绘制三维网格图
                    # ax.plot_wireframe(x, y, abs_Hdd[0,:,:].cpu().detach().numpy(), color='blue')

                    # # 设置标签
                    # ax.set_xlabel('X Axis')
                    # ax.set_ylabel('Y Axis')
                    # ax.set_zlabel('Z Axis')
                    # ax.set_title('3D Wireframe Plot')

                    # # 显示图形
                    # plt.savefig('channel.png')
                    # plt.close()
                    # print(lp[0,:],vp[0,:])
                    # print(lp.flatten().shape)
                    # Gather channel estimation data
                    
                    
                    # pathGains = topk_vals.view(Hdd.size(0), top_k)
                    # pathDelays = lp
                    # pathDopplers = vp - 17
                    
                    # chanEst = {
                    # 'pathGains': pathGains,
                    # 'pathDelays': pathDelays,
                    # 'pathDopplers': pathDopplers
                    # }

                    
                    # Form G matrix using channel estimates
                    # G = getG(self.opt.N_pilot, self.opt.S, chanEst, self.opt.K, None,device=self.device).to(torch.complex64)#.cpu()
                    # # print(G.shape)
                    # chOut = out_sig.reshape(N,-1)
                    # # chOut = out_pilot.reshape(N,-1)
                    # # Perform LMMSE equalization
                    # rxWindow = chOut[:, :G.size(1)]#.cpu()  # Adjusting the sample size if needed
                    # G_H = G.conj().transpose(-1, -2)#.cpu()  # Hermitian transpose of G
                    # # print(G.shape)
                    # # print(G_H.shape)
                    # t = G_H @ G + noise_pwr * torch.eye(G.size(1)).to(G.device)
                    # epsilon = 1e-6  # A small value to ensure numerical stability
                    # # t = t + epsilon * torch.eye(t.size(-1)).to(t.device)
                    
                    # # print(t.shape)
                    # # print(G_H.dtype)
                    # # print(rxWindow.shape)
                    # # print((G_H @ rxWindow.unsqueeze(-1)).dtype)
                    # t2 = G_H.to(t.dtype) @ rxWindow.unsqueeze(-1).to(t.dtype)
                    # # print(t2.shape)
                    # # self.rx = torch.linalg.solve(t, G_H.to(t.dtype) @ rxWindow.unsqueeze(-1).to(t.dtype)).squeeze().reshape(N,self.opt.P,self.opt.S,self.opt.M)
                    
                    # self.rx = torch.linalg.solve(t,t2).reshape(N,self.opt.P,self.opt.S,self.opt.M)
                    # print(Hdd.shape)
                    # print(out_sig.shape)
                    # Hdd = torch.fft.fft2(Hdd)
                    # print(Hdd.shape)
                    out_sig = torch.fft.fft2(out_sig)
                    self.rx = Hdd.conj()*out_sig.squeeze()/(torch.abs(Hdd)**2+noise_pwr)
                    self.rx = torch.fft.ifft2(self.rx)
                    self.rx = self.rx.reshape(N,self.opt.P,self.opt.S,self.opt.M)
                    # x = np.linspace(0,31,32)
                    # y = np.linspace(0,11,12)
                    # x,y = np.meshgrid(y,x)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    
                    # # 绘制三维网格图
                    # ax.plot_wireframe(x, y, torch.abs(self.rx[0,0,:,:]).cpu().detach().numpy(), color='blue')

                    # # 设置标签
                    # ax.set_xlabel('X Axis')
                    # ax.set_ylabel('Y Axis')
                    # ax.set_zlabel('Z Axis')
                    # ax.set_title('3D Wireframe Plot')

                    # # 显示图形
                    # plt.savefig('rec.png')
                    # plt.close()
                    # print(self.rx.shape)
                    

                r1 = torch.view_as_real(self.rx)
                # dec_in = r1.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W).float() 
                 
                #S32 M12
                if self.opt.S == 32:
                    dec_in = r1.contiguous().permute(0,1,3,4,2).contiguous().view(N, -1, H, W).float()  
                elif self.opt.S == 16:
                    dec_in = r1.permute(0,1,3,4,2).contiguous().view(N, -1, H, W).float() 
                self.fake = self.netG(dec_in)

            elif self.opt.feedforward == 'EXPLICIT-RES':
                if self.opt.is_pm:
                    self.H_est = self.channel_estimation(out_pilot, noise_pwr) 
                    
                    sub11 = torch.view_as_real(self.ofdm.pilot).repeat(N,1,1,1,1)
                    sub12 = torch.view_as_real(out_pilot)
                    sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0,1,3,2,4).contiguous().view(N, -1, H, W).float()
                    sub1_output = self.netS1(sub1_input).view(N, self.opt.P, 1, 2, self.opt.S).permute(0,1,4,2,3)
                self.H_est = self.channel_estimation(out_pilot, noise_pwr) 
                
                # print("est1 shape",self.H_est.shape)
                sub11 = torch.view_as_real(self.ofdm.pilot).repeat(N,1,1,1,1)
                sub12 = torch.view_as_real(out_pilot)
                # sub11 = torch.cat((sub11,torch.zeros(N,1,1,32-self.opt.M,2).to(self.device)),3)
                # sub12 = torch.cat((sub12,torch.zeros(N,1,self.opt.N_pilot,32-self.opt.M,2).to(self.device)),3)
                # print(sub11.shape)
                # print(sub12.shape)
                # sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, 3, 4).float()
                # # print(sub1_input.shape)
                # sub1_output = self.netS1(sub1_input).view(N, self.opt.P, 1, 2, self.opt.M).permute(0,1,2,4,3)
                
                # print(sub11.shape)
                # print(sub12.shape)
                self.H_est = self.H_est.permute(0,1,3,2)
                sub11 = sub11.contiguous().view(N,self.opt.P , self.opt.S,1,2)

                sub1_input = torch.cat((sub11,sub12), 3).contiguous().permute(0,1,3,2,4).contiguous().view(N, -1, 4, 4).float()
                sub1_output = self.netS1(sub1_input).view(N, self.opt.P, 1, 2, self.opt.S).permute(0,1,4,2,3)
                # print("est shape",self.H_est.shape)
                # print("sub1shape",sub1_output.shape)
                self.H_est = self.H_est + torch.view_as_complex(sub1_output.contiguous())

                # print("est shape",self.H_est.shape)

                self.rx = self.equalization(self.H_est, out_sig, noise_pwr)  
                sub21 = torch.view_as_real(self.H_est)
                sub22 = torch.view_as_real(out_sig)
                # print(sub21.shape)
                # print(sub22.shape)
                # sub2_input = torch.cat((sub21, sub22), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, 3, 4).float()
                # sub2_output = self.netS2(sub2_input).view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
                # sub2_output = self.netS2(sub2_input).view(N, self.opt.P, self.opt.M, 2, self.opt.S).permute(0,1,3,4,2)

                # sub21 = sub21.squeeze(1).unsqueeze(-2)
                # print(sub21.shape)
                # print(sub22.shape)
                sub2_input = torch.cat((sub21, sub22), 3).contiguous().permute(0,1,3,2,4).contiguous().view(N, -1, 4, 4).float()
                sub2_output = self.netS2(sub2_input).view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)

                self.rx = self.rx + torch.view_as_complex(sub2_output.contiguous())

                # dec_in = torch.view_as_real(self.rx).permute(0,1,2,4,3).contiguous().view(latent.shape).float()
                dec_in = torch.view_as_real(self.rx).permute(0,1,3,4,2).contiguous().view(latent.shape).float()

                self.fake = self.netG(dec_in)
                
            elif self.opt.feedforward in ['EXPLICIT-EQ','JSCC'] :
                # r3 = torch.view_as_real(out_sig)
                # r2 = torch.view_as_real(out_pilot)
                # rnn_input = torch.cat((r3, r1, r2), 3).contiguous().view(N, self.opt.S,-1).float()
                

                r3 = torch.view_as_real(out_sig).contiguous().view(N,1,self.opt.S,self.opt.M*2)
                # r3 = torch.view_as_real(self.tx_c).contiguous().view(N,1,self.opt.S,self.opt.M*2)
                if self.opt.is_pm:
                    r2 = torch.view_as_real(out_pilot).contiguous().view(N,1,self.opt.N_pilot,self.opt.M*2)
                    rnn_input = torch.cat((r3, r1, r2), 2).contiguous().view(N, self.opt.S+self.opt.N_pilot*2,self.opt.M*2).float()
                else:
                    r2 = torch.view_as_real(out_pilot).contiguous().view(N,1,self.opt.S,-1)
                    r2 = torch.cat((r2[:,:,:,:self.opt.N_pilot].repeat(1,1,1,int(self.opt.M/self.opt.N_pilot)),r2[:,:,:,self.opt.N_pilot:].repeat(1,1,1,int(self.opt.M/self.opt.N_pilot))),-1)

                    rnn_input = torch.cat((r3, r1, r2), 2).contiguous().view(N, self.opt.S*3,self.opt.M*2).float()

                if self.opt.is_t :

                    self.rnn_output = self.netT(rnn_input)
                else:

                    self.rnn_output = rnn_input

                # dec_in = self.rnn_output.contiguous().view(N, self.opt.P, self.opt.S,self.opt.M,2)[:,:,:,:self.opt.M,:].contiguous()
                
                if self.opt.is_pm:
                    dec_in = self.rnn_output.contiguous().view(N, self.opt.P,self.opt.S+self.opt.N_pilot*2,self.opt.M,2)[:,:,:self.opt.S,:,:].contiguous()
                else:
                    dec_in = self.rnn_output.contiguous().view(N, self.opt.P,self.opt.S*3,self.opt.M,2)[:,:,:self.opt.S,:,:].contiguous()
                # N = self.opt.batch_size
                # 
                # dec_in = r3.view(N, self.opt.P,self.opt.S,self.opt.M,2)
                if self.opt.is_cm:
                # s32 m 12
                    dec_in = dec_in.permute(0,1,3,4,2).contiguous().view(N, C, H, W).float()
                else:
                # S 6 M 64 
                # s 12 m 32
                # dec_in = dec_in.contiguous().view(N, C, H, W)
                #s 12 m 32
                    dec_in = dec_in.permute(0,1,2,4,3).contiguous().view(N, C, H, W).float()

                # print(dec_in.shape,t.shape)
                # print(torch.equal(dec_in,t))
                
                # dec_in = r3.view(N, self.opt.P,-1,self.opt.M,2).permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
                self.fake = self.netG(dec_in,v=velocity)


        


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        
        _, pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_data = self.real_B
        _, pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.opt.gan_mode == 'wgangp':
            penalty, grad = networks.cal_gradient_penalty(self.netD, real_data, self.fake.detach(), self.device, type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
            self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.gan_mode != 'none':
            feat_fake, pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            if self.is_Feat:
                feat_real, pred_real = self.netD(self.real_B)
                self.loss_G_Feat = 0
                
                for j in range(len(feat_real)):
                    self.loss_G_Feat += self.criterionFeat(feat_real[j].detach(), feat_fake[j]) * self.opt.lambda_feat
            else:
                self.loss_G_Feat = 0     
        else:
            self.loss_G_GAN = 0
            self.loss_G_Feat = 0 

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2

            
        diff = 0
        # diff = deconv_orth_dist(self.netE.module.model_down[4].weight, stride=2) + deconv_orth_dist(self.netE.module.model_down[7].weight, stride=2) #+deconv_orth_dist(self.netG.module.model[0].weight, stride=2) + deconv_orth_dist(self.netG.module.model[3].weight, stride=2) 
        # diff += deconv_orth_dist(self.netE.module.res1.conv_block[1].weight, stride=1) 
        # diff+= deconv_orth_dist(self.netE.module.res2.conv_block[1].weight, stride=1)
        # diff += deconv_orth_dist(self.netE.module.projection.weight, stride=1) #+ deconv_orth_dist(self.netG.module.conv1.weight, stride=1)
        
        # diff += deconv_orth_dist(self.netG.module.res1.conv_block[1].weight, stride=1) + deconv_orth_dist(self.netG.module.res2.conv_block[1].weight, stride=1)
        # diff+= orth_dist(self.netE.module.model_down[1].weight,stride=1)
        # diff = orth_dist(self.netE.module.model_down[4].weight, stride=2) + orth_dist(self.netE.module.model_down[7].weight, stride=2)
        # diff+=orth_dist(self.netE.module.projection.weight, stride=1)
        # diff+= orth_dist(self.netE.module.res1.conv_block[1].weight, stride=1) + orth_dist(self.netE.module.res2.conv_block[1].weight, stride=1)
        # diff+=orth_dist(self.netG.module.conv1.weight, stride=1)
        # diff+= orth_dist(self.netG.module.res1.conv_block[1].weight, stride=1) + orth_dist(self.netG.module.res2.conv_block[1].weight, stride=1)
        self.loss_OR = diff * self.opt.lambda_or
        # self.loss_OR = 0

        self.loss_PAPR = torch.mean(self.PAPR_cp) * self.opt.lambda_papr
        # self.loss_PAPR = 0
        if self.opt.feedforward == 'EXPLICIT-RES':
            # print("ht", self.H_true.shape)
            # print("he", self.H_est.shape)
            self.loss_CE = self.criterionL2(torch.view_as_real(self.H_true.squeeze()), torch.view_as_real(self.H_est.squeeze())) * self.opt.lambda_ce
            # self.loss_CE = 0
            self.loss_EQ = self.criterionL2(torch.view_as_real(self.rx).float(), torch.view_as_real(self.tx_c).float()) * self.opt.lambda_eq

        elif self.opt.feedforward in ['EXPLICIT-EQ','JSCC'] :
            
            if self.opt.is_t:

                self.loss_EQ = self.criterionL2(self.rnn_output[:,:self.opt.S,:], self.rnn_ori[:,:self.opt.S,:]) * self.opt.lambda_eq
            else:
                self.loss_EQ = 0
            self.loss_CE = 0
        else:
            # print("ht", self.H_true.shape)
            # print("he", self.H_est.shape)
            # self.loss_CE = self.criterionL2(torch.view_as_real(self.H_true.squeeze()), torch.view_as_real(self.H_est.squeeze())) * self.opt.lambda_ce
            self.loss_CE = 0
            self.loss_EQ = 0
            # self.loss_EQ = self.criterionL2(torch.view_as_real(self.rx).float(), torch.view_as_real(self.tx_c).float()) * self.opt.lambda_eq

        # print(self.loss_G_L2)
        # print(self.loss_CE)
        # print(self.loss_EQ)
        if self.opt.feedforward == 'EXPLICIT-RES':
            self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2 + self.loss_PAPR + self.loss_OR + self.loss_EQ
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2 + self.loss_PAPR + self.loss_OR #+ self.loss_EQ #+ self.loss_CE  

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan_mode != 'none':
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def channel_estimation(self, out_pilot, noise_pwr):
        if self.opt.CE == 'LS':
            H_est = LS_channel_est(self.ofdm.pilot, out_pilot)
        elif self.opt.CE == 'LMMSE':
            H_est = LMMSE_channel_est(self.ofdm.pilot, out_pilot, self.opt.M*noise_pwr)
        elif self.opt.CE == 'TRUE':
            H_est = self.H_true.unsqueeze(2).to(self.device)
        else:
            raise NotImplementedError('The channel estimation method [%s] is not implemented' % CE)

        return H_est

    def equalization(self, H_est, out_sig, noise_pwr):
        # Equalization
        if self.opt.EQ == 'ZF':
            rx = ZF_equalization(H_est, out_sig)
        elif self.opt.EQ == 'MMSE':
            rx = MMSE_equalization(H_est, out_sig, self.opt.M*noise_pwr)
        elif self.opt.EQ == 'None':
            rx = None
        else:
            raise NotImplementedError('The equalization method [%s] is not implemented' % CE)
        return rx
