import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePathEulerSolver(nn.Module):
    def __init__(self, net, N):
        super(SimplePathEulerSolver, self).__init__()
        '''alpha(t) = 1 - t, beta(t) = t'''
        self.net = net
        self.N = N
        self.delta_t = 1/N
        self.t_span = torch.linspace(0, 1, N+1)
    
    @torch.no_grad()
    def forward(self, x0):
        xt = x0
    
        for n in range(self.N):
            interpolants = torch.ones_like(x0) * self.t_span[n].item()
            out = self.net(x=xt, interpolants=interpolants)
            g0, g1 = out[:, :3, :, :], out[:, 3:, :, :]
            bt = g1 - g0
            xt = xt + bt * self.delta_t
        
        return xt


class AdaptivePathEulerSolver(nn.Module):
    def __init__(self, adaptive_interpolants, N):
        super(AdaptivePathEulerSolver, self).__init__()
        self.ai = adaptive_interpolants
        self.N = N
        self.t_span = torch.linspace(0, 1, N+1)
    
    @torch.no_grad()
    def forward(self, x0):
        xt = x0

        # get net_alpha outputs
        net_alpha_out = self.ai.net_alpha(x=x0)
        
        for n in range(self.N):
            time = torch.ones(x0.shape[0]).type_as(x0) * self.t_span[n].item()
            time_future = torch.ones(x0.shape[0]).type_as(x0) * self.t_span[n + 1].item()

            # get trigonometric spans
            sin_span, cos_span = self.ai.get_trigonometric_spans(time)

            # get alpha tildas and derivatives
            alpha_0_tilda, alpha_1_tilda = self.ai.get_alpha_tildas(net_alpha_out, time, sin_span)
            alpha_0_tilda_future, alpha_1_tilda_future = self.ai.get_alpha_tildas(net_alpha_out, time_future, sin_span)
            alpha_0_tilda_derivative, alpha_1_tilda_derivative = self.ai.get_alpha_tilda_derivatives(
                net_alpha_out, sin_span, cos_span)

            # get alpha hats
            _, alpha_1 = self.ai.get_alpha_hats(alpha_0_tilda, alpha_1_tilda)
            _, alpha_1_future = self.ai.get_alpha_hats(alpha_0_tilda_future, alpha_1_tilda_future)
            delta_alpha = alpha_1_future - alpha_1

            # get derivatives of alpha hats
            alpha_0_derivative, alpha_1_derivative = self.ai.get_alpha_hat_derivatives(
                alpha_0_tilda, alpha_1_tilda, alpha_0_tilda_derivative, alpha_1_tilda_derivative)

            # get gs
            net_g_out = self.ai.net_g(x=xt, interpolants=alpha_1)
            
            # caculate bt with g and alphas
            bt = alpha_0_derivative * net_g_out[:, :3, : ,:] + alpha_1_derivative * net_g_out[:, 3:, :, :]
            
            # calculate next xt
            xt = xt + bt * delta_alpha
        
        return xt