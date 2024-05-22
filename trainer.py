import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Trainer(nn.Module):
    def __init__(self, net, mode, scale, initial_interpolants_length=51, gauss_kernel_size=18, gauss_sigma=4.0):
        super(Trainer, self).__init__()

        # model
        self.net = net

        # Create a Gaussian kernel
        gauss_kernel = torch.Tensor(cv2.getGaussianKernel(gauss_kernel_size, gauss_sigma))
        gauss_kernel = gauss_kernel * gauss_kernel.transpose(0, 1)
        gauss_kernel = gauss_kernel.expand(3, 1, gauss_kernel.shape[0], gauss_kernel.shape[1])
        self.gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
        self.padding = gauss_kernel_size // 2
        
        # etc
        self.mode = mode
        self.scale = scale
        self.initial_interpolants_length = initial_interpolants_length
        self.mse_loss = nn.MSELoss()

    @torch.no_grad()
    def get_gaussian_term(self, img):
        # initialize
        gaussian_term = torch.rand(img.shape[0], 3, self.initial_interpolants_length, self.initial_interpolants_length).type_as(img)

        # Apply the filter
        gaussian_term = F.conv2d(gaussian_term, self.gauss_kernel.type_as(img), padding=self.padding, groups=3)
        gaussian_term = gaussian_term[:, :, 10:-10, 10:-10]
        gaussian_term = (gaussian_term - gaussian_term.min()) / (gaussian_term.max() - gaussian_term.min())
        gaussian_term = gaussian_term * 2 - 1
        gaussian_term = gaussian_term * self.scale
        
        return gaussian_term

    def forward(self, img:Tensor):
        # get interpolants
        if self.mode == 'si':
            interpolants = torch.ones_like(img) * torch.rand(img.shape[0], 1, 1, 1).type_as(img)
        elif self.mode == 'ami_gni':
            interpolants = torch.ones_like(img) * torch.rand(img.shape[0], 1, 1, 1).type_as(img)
            interpolants = interpolants + torch.randn_like(img) * self.scale
            interpolants = torch.clamp(interpolants, min=0.0, max=1.0)
        elif self.mode == 'ami_lpfi':
            interpolants = self.get_gaussian_term(img)
            interpolants = interpolants + torch.ones_like(img) * torch.rand(img.shape[0], 1, 1, 1).type_as(img)
            interpolants = torch.clamp(interpolants, min=0.0, max=1.0)

        # get x0, x1 and noise term
        x0, x1 = torch.randn_like(img), img

        # interpolate
        x_interpolated = interpolants * x1 + (1 - interpolants) * x0

        # get g values
        out = self.net(x=x_interpolated, interpolants=interpolants)
        g0, g1 = out[:, :3, :, :], out[:, 3:, :, :]

        # calculate loss
        temp_1 = torch.norm(g0, p=2, dim=(1, 2, 3))**2
        temp_2 = 2 * torch.sum(x0 * g0, dim=(1, 2, 3))

        loss_0 = torch.mean(torch.norm(g0, p=2, dim=(1, 2, 3))**2 - 2 * torch.sum(x0 * g0, dim=(1, 2, 3)))
        loss_1 = torch.mean(torch.norm(g1, p=2, dim=(1, 2, 3))**2 - 2 * torch.sum(x1 * g1, dim=(1, 2, 3)))
        loss = loss_0 + loss_1

        return loss, loss_0.item(), loss_1.item()