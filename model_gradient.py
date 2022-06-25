# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.nn.parallel
import torch.nn.parallel
import torch.nn.modules
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from collections import OrderedDict
import pytorch_ssim
from exclusion_loss import my_compute_gradient,compute_gradient_img_my,Scharr_demo
from torchvision.utils import save_image

def swish(x):
    return x * F.sigmoid(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)
#FeatureExtractor =torch.nn.DataParallel(FeatureExtractor,device_ids=[0,1])

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        m['bn1'] = nn.BatchNorm2d(n)
        m['ReLU1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(n, n, k, stride=s, padding=1)
        m['bn2'] = nn.BatchNorm2d(n)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))


    def forward(self, x):
#        y = swish(ReLU1(bn1(conv1(x))))
#        y = swish(m[self.ReLU1]((m[self.bn1](m[self.conv1](x)))))



        out = self.group1(x) + x



#        out = m[self.bn2](m[self.conv2](y)) + x
        out = self.relu(out)
        return out
#        return self.bn2(self.conv2(y)) + x


class Generator(nn.Module):
    def __init__(self, n_residual_blocks):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        #self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(2, 64, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)


        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)



        #for i in range(self.upsample_factor//5):
        #    self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))


        self.conv6 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #print('x size:',x.size())
        img = x[:,0,:,:].squeeze(1)
        #print(type(img))
        #print('cpu:',img.cpu().numpy().shape)

        #im_gred = compute_gradient_img_my(img.cpu().numpy())
        #im_gred = Variable(torch.from_numpy(im_gred)).cuda()
        #save_image(im_gred.data, './output_train/' + 'gradient_np.bmp')
        im_gred = Scharr_demo(img.cpu().numpy())
        im_gred = Variable(torch.from_numpy(im_gred)).float().cuda()/255
        #save_image(im_gred.data, './output_train/' + 'gradient_CV.bmp')
        im_gred = im_gred.repeat(1,64,1,1)
        #print(im_gred[:,0,:,:])
        #print(im_gred[:,1,:,:])
        #print('im_gred:',type(im_gred))
        #im_gred = my_compute_gradient(img)
        #im_gred = torch.from_numpy(im_gred)
        x = swish(self.relu1(self.bn1(self.conv1(x))))
        x = swish(self.relu2(self.bn2(self.conv2(x))))
        x = swish(self.relu3(self.bn3(self.conv3(x))))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
            y = y+ im_gred*1

        x = swish(self.relu4(self.bn4(self.conv4(y)))) + x
        x = swish(self.relu5(self.bn5(self.conv5(x))))
        #return self.conv6(x)
        return self.sigmoid(self.conv6(x))
#Generator = nn.DataParallel(Generator()).cuda()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
#generator =torch.nn.DataParallel(Generator,device_ids=[0,1],dim =1)
#discriminator =torch.nn.DataParallel(Discriminator,device_ids=[0,1])
#feature_extractor =torch.nn.DataParallel(FeatureExtractor,device_ids=[0,1],dim =1)
