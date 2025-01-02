# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from collections import OrderedDict
import random
import time
from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
from torch.utils.data.sampler import SubsetRandomSampler
# from pytorch_lightning import seed_everything
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=526):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 526

# seed_everything(seed=seed)
# Extract the options
opt = TrainOptions().parse()


if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # # dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
    #                                          shuffle=True, num_workers=2, drop_last=True)
    
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                         download=True, transform=transform)
    # test_dataset = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                            #  shuffle=False, num_workers=2,drop_last=True)
 
    valid_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform,
    )

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataset = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True,worker_init_fn=np.random.seed(seed),drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, sampler=valid_sampler,
        num_workers=2, pin_memory=True,worker_init_fn=np.random.seed(seed),drop_last=True
    )
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
    print('#valid images = %d' % len(valid_loader))
elif opt.dataset_mode == 'CelebA':
    opt.dataroot='./data'
    # opt.dataroot='./img_align_celeba'
    opt.size = 128
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(root='./celeba',
                                            transform=transform)
    # dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                            #  shuffle=True, num_workers=2, drop_last=True)
    
    valid_dataset = torchvision.datasets.ImageFolder(
        root='./celeba', transform=transform
    )

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataset = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True,worker_init_fn=np.random.seed(seed),drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, sampler=valid_sampler,
        num_workers=2, pin_memory=True,worker_init_fn=np.random.seed(seed),drop_last=True
    )
    dataset_size = len(dataset)
else:
    raise Exception('Not implemented yet')


model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
total_iters = 0                # the total number of training iterations
# print(model.netE.module.model_down[4].weight.data)
# for name, param in model.named_parameters():
#     print(f"Parameter {name}:")
#     print(param.data)
#     print()

maxvalid = 0
noimpepoch = 0
################ Train with the Discriminator
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    valid = 0
    
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        if opt.dataset_mode == 'CIFAR10':
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data[0]
 

        ##test
        opt.V = random.uniform(opt.v_min,opt.v_range)
        # opt.V = 20
        # opt.SNR = random.uniform(5,20)

        model.set_input(input)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size / opt.batch_size, losses)

        # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #     model.save_networks(save_suffix)
        iter_data_time = time.time()
    
    

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

    test_iter = 0
    #test
    model.eval()
    # opt.v_min = 100
    # opt.v_range = 300
    PSNR = []
    valids = []
    for i, data in enumerate(valid_loader):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        
        ##test
        opt.V = opt.v_range

        test_iter += opt.batch_size
        if opt.dataset_mode == 'CIFAR10':
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data[0]

        model.set_input(input) 
        model.forward()
        fake = model.fake
        # loss = model.criterionL2(fake, input.to(model.device)) * opt.lambda_L2
        # losses_t = OrderedDict()
        
        # losses_t['valid'] = float(loss)

        img_gen_numpy = fake.detach().cpu().float().numpy()
        img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        img_gen_int8 = img_gen_numpy.astype(np.uint8) 

        origin_numpy = input.detach().cpu().float().numpy()
        origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        origin_int8 = origin_numpy.astype(np.uint8)
        

        diff = np.mean((np.float64(img_gen_int8)-np.float64(origin_int8))**2, (1,2,3))

        PSNR.append(10*np.log10((255**2)/diff))
                
        
        # print(losses.keys())
        # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
        # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
        # if test_iter %  test_print_frp == 0:
        valid = float(np.mean(PSNR))
        valids.append(valid)
       
    if opt.display_id > 0:
        losses_t = OrderedDict()
        losses_t['valid'] = valid
        visualizer.plot_current_losses_t(epoch, float(test_iter) / len(valid_loader) / opt.batch_size, losses_t)

        # model.save_networks(epoch)

    valid_mean = np.mean(valids)
    print(valid_mean)

    
        
    model.train()
    # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
    # model.save_networks('%s'%opt.tname)
    if  valid_mean > maxvalid :              # cache our model every <save_epoch_freq> epochs
        print(valid_mean,maxvalid)
        maxvalid = valid_mean
        noimpepoch = 0
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('%s'%opt.tname)
    else:
        noimpepoch += 1
        print(noimpepoch)
        if noimpepoch > 150:
            break

    # opt.v_min = 0
    # opt.v_range = 100
