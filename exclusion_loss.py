#-*- coding:utf-8 -*-
#@bao
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

def compute_gradient_img_my(img):
    # input : Image/nparray 0~255
    if isinstance(img, Image.Image):
        img = np.array(img)
        #print(img.shape)
    #dx, dy = np.gradient(img[0,:,:].squeeze() / 1, edge_order=1)
    #dxy = (dx+dy)/2
    #print(dxy)
    img = img*255
    print(np.shape(img))
    b,h,w = np.shape(img)
    img_grad = np.zeros(((b,1,h,w)))
    #print((img_grad.size()))
    for i in range(len(img)):
        #im = img[i,0,:,:].squeeze()
        dx, dy = np.gradient(img[i,:,:].squeeze() / 1, edge_order=1)
        dxy = (dx+dy)/2
        img_grad[i,0,:,:]=dxy


    #print(batch_s)
    #im = img[0,:,:].squeeze()
    #print(type(im))
    #im2 = im
    #im = (im*255)#.astype(np.uint8)
    #im = Image.fromarray(im)
    #plt.show(im)

    #dx, dy = np.gradient(im / 1, edge_order=1)

    #dx, dy = compute_gradient(img)
    #img = (np.sqrt(dx ** 2 + dy ** 2) * 1)
    #print(type(img))
    #Image.fromarray(dx*255).show()
    #Image.fromarray(dy*255).show()
    #img = ((dx * 0.5+ dy * 0.5) * 255).astype(np.uint8)
    #img = Image.fromarray(img)
    #img.show()

    #if isinstance(img,np.ndarray):
    #    img = Image.fromarray(img)
    return img_grad

import cv2 as cv
#Scharr算子(Sobel算子的增强版，效果更突出)
def Scharr_demo(img):
    img = img*255
    b,h,w = np.shape(img)
    img_grad = np.zeros(((b,1,h,w)))
    #print((img_grad.size()))
    for i in range(len(img)):
        image = img[i,:,:].squeeze()
        grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #对x求一阶导
        grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #对y求一阶导
        gradxy = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        img_grad[i,0,:,:]=gradxy

    return img_grad

    '''image = img[0,:,:].squeeze()
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradxy = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    print(gradxy)'''
    '''gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  #x方向上的梯度
    cv.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)'''
'''src = cv.imread('E:/imageload/liu.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
Scharr_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()'''

def my_compute_gradient(img):
    #tensor:(N, C, H, W)
    # gradx = img[1:, :, :] - img[:-1, :, :]
    # grady = img[:, 1:, :] - img[:, :-1, :]
    N,C,H,W = img.shape()
    gradx = img[:, :, 1:, :] - img[:, :, :-1, :,] # gradient on width
    grady = img[:, :, :, 1:] - img[:, :, :, :-1] #gradient on height
    img_gred = gradx**2 + grady**2

    return img_gred


def compute_gradient_img(img):
    # input : Image/nparray 0~255
    if isinstance(img, Image.Image):
        img = np.array(img)
        print(img.shape)
    dx, dy = np.gradient(img / 255, edge_order=1)
    #dx, dy = compute_gradient(img)
    img = (np.sqrt(dx ** 2 + dy ** 2) * 255).astype(np.uint8)

    if isinstance(img,np.ndarray):
        img = Image.fromarray(img)
    return img

def compute_gradient(img):
    #tensor:(N, C, H, W)
    # gradx = img[1:, :, :] - img[:-1, :, :]
    # grady = img[:, 1:, :] - img[:, :-1, :]
    gradx = img[:, :, 1:, :] - img[:, :, :-1, :,] # gradient on width
    grady = img[:, :, :, 1:] - img[:, :, :, :-1] #gradient on height

    return gradx, grady

def compute_exclusion_loss_level(img1, img2, level=1):
    '''
    exclusion loss : enforces separation of the transmission and reﬂection layers in the gradient domain.
    minimize the correlation betweeen two layers in gradient domain
    :param img1:  transmission layer image
    :param img2:  reflection layer image
    :param level: channels
    :return: gradx_loss, grady_loss #loss on x ,y axis respectively
    '''
    gradx_loss = []
    grady_loss = []

    for l in range(level):# for each channel:
        print(l)
        # print('img1/',img1[:,l,:,:].unsqueeze(1).size())
        gradx1, grady1 = compute_gradient(img1[:,l,:,:].unsqueeze(1)) # gradient on width(x),height(y) transmission layer
        gradx2, grady2 = compute_gradient(img2[:,l,:,:].unsqueeze(1)) # gradient on width(x),height(y) reflection layer
        print(gradx1)
        print(grady1)

        alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2)) #trans/ref ----lamda R --x
        alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2)) #trans/ref ----lamda R --y
        print(alphax)
        print(alphay)

        gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
        grady1_s = (torch.sigmoid(grady1) * 2) - 1
        print(gradx1_s)
        print(grady1_s)

        gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1
        print(gradx2_s)
        print(grady2_s)



        #print(torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2)).size())
        # gradx_tensor = torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2))
        # print(gradx_tensor)
        # print('gradx_tensor',gradx_tensor[:,-1,-1,-1].size()) #gradx_tensor torch.Size([2])
        # grad_mean_x = torch.mean(gradx_tensor[:,-1,-1,-1])
        # gradx_loss.append(grad_mean_x)
        gradx_loss.append(
            torch.mean(torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2))[:,-1,-1,-1]) ** 0.25)
        grady_loss.append(
            torch.mean(torch.mul(torch.pow(grady1_s,2), torch.pow(grady2_s,2))[:,-1,-1,-1])** 0.25)

    avepool = torch.nn.AvgPool2d(kernel_size=(2,2), stride=2)#(N, C, H, W)
    img1 = avepool(img1)
    img2 = avepool(img2)

    return gradx_loss, grady_loss


def compute_exclusion_loss(img1, img2, level=1):
    '''
    exclusion loss : enforces separation of the transmission and reﬂection layers in the gradient domain.
    minimize the correlation betweeen two layers in gradient domain
    :param img1:  transmission layer image
    :param img2:  reflection layer image
    :param level: channels
    :return: gradx_loss, grady_loss #loss on x ,y axis respectively
    '''
    gradx_loss = []
    grady_loss = []

    for l in range(level):# for each channel:
        #print(l)
        # print('img1/',img1[:,l,:,:].unsqueeze(1).size())
        gradx1, grady1 = compute_gradient(img1) # gradient on width(x),height(y) transmission layer
        gradx2, grady2 = compute_gradient(img2) # gradient on width(x),height(y) reflection layer
        #print(gradx1)
        #print(grady1)

        alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2)) #trans/ref ----lamda R --x
        alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2)) #trans/ref ----lamda R --y
        #print(alphax)
        #print(alphay)

        gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
        grady1_s = (torch.sigmoid(grady1) * 2) - 1
        #print(gradx1_s)
        #print(grady1_s)

        gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1
        #print(gradx2_s)
        #print(grady2_s)



        #print(torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2)).size())
        # gradx_tensor = torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2))
        # print(gradx_tensor)
        # print('gradx_tensor',gradx_tensor[:,-1,-1,-1].size()) #gradx_tensor torch.Size([2])
        # grad_mean_x = torch.mean(gradx_tensor[:,-1,-1,-1])
        # gradx_loss.append(grad_mean_x)
        gradx_loss.append(
            torch.mean(torch.mul(torch.pow(gradx1_s,2), torch.pow(gradx2_s,2))[:,-1,-1,-1]) ** 0.25)
        grady_loss.append(
            torch.mean(torch.mul(torch.pow(grady1_s,2), torch.pow(grady2_s,2))[:,-1,-1,-1])** 0.25)

        avepool = torch.nn.AvgPool2d(kernel_size=(2,2), stride=2)#(N, C, H, W)
        img1 = avepool(img1)
        img2 = avepool(img2)

    return gradx_loss, grady_loss

 # Gradient loss
def exclusion_loss(transmission_layer,reflection_layer,level=3,issyn = True):
    #issyn is a bool condition see original code
    # ----if magic < 0.7:  # choose from synthetic dataset
     #               is_syn = True
    #print('level',level)
    loss_gradx,loss_grady=compute_exclusion_loss(transmission_layer,reflection_layer,level)
    #print(loss_gradx)
    #print(loss_grady)
    loss_gradxy = sum(loss_gradx)/float(level) + sum(loss_grady)/float(level)
    #loss_gradxy =(loss_gradxy/2.0 if issyn else 0) ##loss_grad=tf.where(issyn,loss_gradxy/2.0,0)

    # @bao: log(1/(original loss))and ignore issyn for image fusion task
    #print(1/(loss_gradxy / 2.0))
    loss_gradxy = torch.log(1/(loss_gradxy / 2.0))
    #print('loss_gradxy',loss_gradxy)
    return loss_gradxy



if __name__ == '__main__':
    img1 = Variable(torch.rand(2, 2, 8, 8))
    img2 = Variable(torch.rand(2, 2, 8, 8))
    # print(torch.is_tensor(img1))
    print(img1)
    #assert img1.size()==img2.size()
    level = img1.size(1)
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    # im = Image.open('./train_real_epoch_000.png').convert('L').crop([1, 1, 200, 200])
    # # fig 1
    # im.show()
    # # fig 2
    # compute_gradient_img(im).show()

    print(exclusion_loss(img1, img2,level,True))
    #print(pytorch_compute_exclusion_loss(img1,img2,1,2))