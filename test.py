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
# from tensorboard_logger import configure, log_value
import torchvision.utils as vutils
from model_gradient import Generator
#import model_li
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
#GPUID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#model =nn.DataParallel(models)
parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size') # param
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
Loss_list = []
PSNR_list = []

opt = parser.parse_args()
opt.out = './output/'
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass
root_tr = './'
root_tt = './test_dataset/TNO_'

def default_loader(path):
    return Image.open(path).convert('L')

class Train_Dataset(Dataset):
    def __init__(self, img_root, txt_name, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt_name, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            input = words[0]
            im_root_vi = img_root
            im_root_ir = img_root.replace('vi', 'ir')
            im_root_mk = img_root.replace('vi', 'softmk_50')
            imgs.append([im_root_vi + '/' + input, im_root_ir + '/' + input, im_root_mk + '/' + input])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path_vi, path_ir, path_mk = self.imgs[index]
        words = path_vi.split('./test_dataset/TNO_vi/')
        img_name = words[1]
        im_vi = self.loader(path_vi)
        im_ir = self.loader(path_ir)
        im_mk = self.loader(path_mk)
        if self.transform is not None:
            im_vi = self.transform(im_vi)
            im_ir = self.transform(im_ir)
            im_mk = self.transform(im_mk)
        return im_vi, im_ir, im_mk, img_name

    def __len__(self):
        return len(self.imgs)

train_test_data = Train_Dataset(img_root=root_tt+'vi', txt_name=root_tt + 'vi.txt', transform=transforms.Compose([transforms.ToTensor()]))
assert Train_Dataset
dataloader_train_test = DataLoader(dataset=train_test_data, batch_size=1, shuffle=False, num_workers=int(opt.workers))
'''------------------------------------------------------'''

def _load_ckpt(model, ckpt):
    load_dict = {}
    for name in ckpt:
        print(name.split('module.')[1])
        load_dict[name.split('module.')[1]]= ckpt[name]
    model.load_state_dict(load_dict, strict=False)
    return model

generator = Generator(6)  # param
path = './models/train_with_TNO.pth'
generator = _load_ckpt(generator, torch.load(path))
print(generator)
ssim_loss = pytorch_ssim.SSIM()

if opt.cuda:
    generator.cuda()

print(len(dataloader_train_test))

for j, data_tt in enumerate(dataloader_train_test):
    im_vi_t = data_tt[0]
    im_ir_t = data_tt[1]
    im_mk_t = data_tt[2]
    im_name = data_tt[3]
    #word = im_name.split('.')
    if opt.cuda:
        im_vi_t = Variable(im_vi_t.cuda())#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
        im_ir_t = Variable(im_ir_t.cuda())#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1])
        im_mk_t = Variable(im_mk_t.cuda())
        im_mg_t = torch.cat((im_vi_t,im_ir_t),1)
        im_mg_wt = generator(im_mg_t)#data_res_gen = torch.nn.DataParallel(generator(Variable(data_mix)),device_ids=[0,1])
        im_vi_gn = torch.mul(im_mg_wt,im_vi_t)
        im_ir_gn = torch.mul(1-im_mg_wt,im_ir_t)
    else:
        im_vi_t = Variable(im_vi_t)#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
        im_ir_t = Variable(im_ir_t)#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1])
        im_mk_t = Variable(im_mk_t)
        im_mg_t = torch.cat((im_vi_t,im_ir_t),1)
        im_mg_gn_t = generator(im_mg_t)
    im_mg_gn_t = im_vi_gn + im_ir_gn
    print(im_mg_gn_t.type())
    print(im_mg_gn_t.size())
    ssim_val = ssim_loss(im_mg_gn_t, im_vi_t) + ssim_loss(im_mg_gn_t, im_ir_t)
    print('%s + SSIM: %f' % (im_name, ssim_val))
    print(type(im_mg_gn_t))
    cvt_im_gn = torchvision.transforms.functional.to_pil_image(im_mg_gn_t.squeeze(0).detach().cpu())
    print(type(cvt_im_gn))
    cvt_im_gn.save('./output result/' +str(im_name)+'.bmp',quality = 75)

