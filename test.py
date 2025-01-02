# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import lpips
import scipy.io as sio
import models.channel as chan
import shutil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math
import matplotlib.pyplot as plt

def seed_everything(seed=526):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# Extract the options
opt = TestOptions().parse()

# opt.batch_size = 1           # batch size

if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=2,drop_last=True)
    dataset_size = len(dataset)
    print('#testing images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './img_align_celeba'
    opt.load_size = 80
    opt.crop_size = 1
    opt.size = 1
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')

########################################  OFDM setting  ###########################################

output_path = opt.output_path
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
    os.makedirs(output_path+'/OFDM')
    os.makedirs(output_path+'/OTFS')
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(output_path+'/OFDM')
    os.makedirs(output_path+'/OTFS')

PSNR_OFDM = {}
PSNR_OTFS = {}
SSIM_dict = {}
lp = []
# lpips_model = lpips.LPIPS(net='alex')
# for v in range(-10,16,2):
# for v in range(-4,22,2):
    # opt.SNR=v
# for v in range(4,11,1):
#     opt.L = v
for v in range(opt.v_min,opt.v_range,opt.v_step):
    opt.V = v

    res_repeat_psnr = 0
    res_repeat_ssim = 0
    repeat_times = 10
    for t in range(repeat_times):
        print(v)
        print("V",opt.V)
        mods = []
        if(opt.modulation=='both'):
            mods = ['OFDM', 'OTFS']
        else:
            mods.append(opt.modulation)
            # print(mods)
        for mod in mods:
            opt.modulation = mod
            print('mod',opt.modulation)
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.eval()
            
            PSNR_list = []
            SSIM_list = []
            for i, data in enumerate(dataset):
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
        
                start_time = time.time()

                if opt.dataset_mode == 'CIFAR10':
                    input = data[0]
                elif opt.dataset_mode == 'CelebA':
                    input = data['data']
                # print(input.shape)
                model.set_input(input)
                model.forward()
                fake = model.fake
                
                # Get the int8 generated images
                img_gen_numpy = fake.detach().cpu().float().numpy()
                img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                img_gen_int8 = img_gen_numpy.astype(np.uint8) 

                origin_numpy = input.detach().cpu().float().numpy()
                origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                origin_int8 = origin_numpy.astype(np.uint8)
                

                diff = np.mean((np.float64(img_gen_int8)-np.float64(origin_int8))**2, (1,2,3))
                
                PSNR = 10*np.log10((255**2)/diff)    
                # print(np.mean(PSNR)) 
                PSNR_list.append(np.mean(PSNR))

                img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
                origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

                ssim_val = ssim(img_gen_tensor, origin_tensor, data_range=255, size_average=False) # return (N,)
                SSIM_list.append(torch.mean(ssim_val))

                # Save the first sampled image
                save_path = output_path + '/' + mod + '/' + str(i) + '_PSNR_' + str(PSNR[0]) +'_SSIM_' + str(ssim_val[0])+'.png'
                util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

                save_path = output_path + '/' + mod + '/' + str(i) + '.png'
                util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)
                if i%100 == 0:
                    print(i)

                

            

                # 使用LPIPS模型计算相似性
                # similarity_score = lpips_model(img_gen_tensor, origin_tensor).detach().numpy()
                # lp.append(similarity_score)
                
            PSNR = np.mean(PSNR_list)
            SSIM = np.mean(SSIM_list)
            print('PSNR: '+str(PSNR))
            print('SSIM: '+str(SSIM))
            # if mod == 'OFDM':
            #     PSNR_OFDM[v] = PSNR
            # elif mod == 'OTFS':
            res_repeat_psnr += PSNR
            res_repeat_ssim += SSIM
        PSNR_OTFS[v] = res_repeat_psnr / repeat_times
        SSIM_dict[v] = res_repeat_ssim / repeat_times
        # lp_mean = np.mean(lp)
        # print(lp_mean)
# plt.plot(list(PSNR_OFDM.keys()), list(PSNR_OFDM.values()), label='OFDM')
plt.plot(list(PSNR_OTFS.keys()), list(PSNR_OTFS.values()), label='OTFS')
# for v in range(-4,22,2):
# for v in range(-10,16,2):
# for v in range(4,11,1):
for v in range(opt.v_min,opt.v_range,opt.v_step):
    # plt.text(v, PSNR_OFDM[v], f'({v}, {PSNR_OFDM[v]:.2f})', fontsize=6, ha='center', va='bottom')
    plt.text(v, PSNR_OTFS[v], f'({v}, {PSNR_OTFS[v]:.2f})', fontsize=6, ha='center', va='bottom')
plt.xlabel('Velocity m/s')
plt.ylabel('PSNR')
plt.legend()
if opt.fig == '':

    plt.savefig('%sresult.png'%opt.epoch)
    np.save('%s.npy'%opt.epoch,PSNR_OTFS)
    np.save('%sSSIM.npy'%opt.epoch,SSIM_dict)
else:
    plt.savefig('%sresult.png'%opt.fig)
    np.save('%s.npy'%opt.fig,PSNR_OTFS)
    np.save('%sSSIM.npy'%opt.fig,SSIM_dict)
# print('MSE CE: '+str(np.mean(H_err_list)))
# print('MSE EQ: '+str(np.mean(x_err_list)))
