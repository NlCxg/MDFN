#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from math import log10
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
# from tensorboard_logger import configure, log_value
import torchvision.utils as vutils
from model_gradient import Generator
#import model_gradient
from utils import Visualizer
import os
import torch.nn.parallel
from multiprocessing import Process
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as Fun
from torchvision.utils import save_image
import pytorch_ssim
from exclusion_loss import compute_gradient_img,exclusion_loss
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size') # param
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator') # param
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
Loss_list = []
PSNR_list = []
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set  GPU card

opt = parser.parse_args()
opt.out = './output models/'
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass
    '''if torch.cuda.is_available() and not opt.cuda:
        models.cuda()
    Generator = nn.DataParallel(Generator, device_ids=[0,1])
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")'''
    '''if torch.cuda.is_available():
        models.cuda()
        models = torch.nn.DataParallel(models, device_ids=[0,1])
        cudnn.benchmark = True'''
root_tr = './'
root_tt = './'

def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, img_root, txt_name, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt_name, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            input = words[0]
            im_root_vi = img_root
            im_root_ir = img_root.replace('Train_vi', 'Train_ir')
            #im_root_mg = img_root.replace('Train_vi', 'Train_softmg')
            im_root_mk = img_root.replace('Train_vi', 'Train_softmk')

            imgs.append([im_root_vi + '/' + input, im_root_ir + '/' + input, im_root_mk + '/' + input])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path_vi, path_ir, path_mk = self.imgs[index]
        im_vi = self.loader(path_vi)
        im_ir = self.loader(path_ir)
        im_mk = self.loader(path_mk)
        if self.transform is not None:
            im_vi = self.transform(im_vi)
            im_ir = self.transform(im_ir)
            im_mk = self.transform(im_mk)

        return im_vi, im_ir, im_mk

    def __len__(self):
        return len(self.imgs)

train_data = MyDataset(img_root=root_tr+'Patch_Train_vi', txt_name=root_tr + 'train_vi.txt', transform=transforms.Compose([transforms.ToTensor()]))
assert MyDataset
dataloader_train = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
'''------------------------------------------------------'''

generator = Generator(6) #
print(generator)

content_criterion = nn.MSELoss()
IR_L1_Loss = torch.nn.L1Loss(size_average=True)
ssim_loss = pytorch_ssim.SSIM()

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    generator = torch.nn.DataParallel(generator, device_ids=[0])
    content_criterion.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
print(len(dataloader_train))

for epoch in range(100):
    mean_generator_loss = 0.0
    for i, data_tr in enumerate(dataloader_train):
        # Generate data
        im_vi = data_tr[0]    # y
        im_ir = data_tr[1]
        im_mk = data_tr[2]
        batch_size_s = (len(im_vi))
        if opt.cuda:
            im_vi = Variable(im_vi.cuda())#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
            im_ir = Variable(im_ir.cuda())#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1])
            im_mg = torch.cat((im_vi,im_ir),1)
            #print(im_mg.size())
            im_mk = Variable(im_mk.cuda())
            im_mg_wt = generator(im_mg)#data_res_gen = torch.nn.DataParallel(generator(Variable(data_mix)),device_ids=[0,1])
        else:
            im_vi = Variable(im_vi)#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
            im_ir = Variable(im_ir)#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1])
            im_mg = torch.cat((im_vi,im_ir),1)
            im_mk = Variable(im_mk)
            im_mg_wt = generator(im_mg)

        im_vi_gn = torch.mul(im_mg_wt,    im_vi)
        im_ir_gn = torch.mul(1-im_mg_wt,  im_ir)
        im_mg_gn = im_vi_gn + im_ir_gn             # fused images

        generator.zero_grad()
        mask = im_mk>0
        mask = mask.type(torch.FloatTensor)

        loss_1 = content_criterion(torch.mul(im_mg_gn, 1 - mask.cuda()), torch.mul(im_vi, 1 - mask.cuda()))
        loss_2 = content_criterion(torch.mul(im_mg_gn, mask.cuda()), torch.mul(im_ir, mask.cuda()))
        loss_3 = 1-ssim_loss(im_mg_gn, im_vi)
        loss_4 = 1-ssim_loss(im_mg_gn, im_ir)
        loss_5 = exclusion_loss(im_mg_gn, im_vi)

        generator_loss = loss_1 +  loss_2 + 0.035 * (loss_3 + loss_4 )+ 0.01* loss_5  # 1012new

        mean_generator_loss += generator_loss
        generator_loss.backward()
        optim_generator.step()
        save_image(im_mg_gn.data, './output/' + '_mg_test.png')
        print('Epoch: [%d], Generator train loss is: [%d], %f | %f | %f | %f | %f | %f' % (epoch, i, generator_loss, loss_1, loss_2, loss_3, loss_4, loss_5))
    save_name = str(epoch) + '_Gradient_Fusion.pth'
    torch.save(generator.state_dict(), '%s/%s' % (opt.out, save_name))
    print('Generator Mean loss is: [%d], %f' % (epoch, mean_generator_loss/i))
