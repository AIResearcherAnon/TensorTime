import torch
import torch.nn as nn


class AdaptiveMultidimensionalPath(nn.Module):
    def __init__(
        self, net_g, net_alpha, discriminator,
        optimizer_net_alpha, optimizer_discriminator,
        integration_steps,
        spatial_dimension): 
        super(AdaptiveMultidimensionalPath, self).__init__()

        # model
        self.net_g = net_g
        self.net_alpha = net_alpha
        self.discriminator = discriminator
        self.M = net_alpha.n_classes // 6

        # optimizer
        self.optimizer_net_alpha = optimizer_net_alpha
        self.optimizer_discriminator = optimizer_discriminator

        # etc
        self.time_span = torch.linspace(0, 1, integration_steps+1)
        self.integration_steps = integration_steps
        self.spatial_dimension = spatial_dimension
        self.bceloss = nn.BCELoss()
    
    def get_trigonometric_spans(self, time):
        # preparing spans
        m_span = torch.arange(1, self.M + 1).unsqueeze(0).type_as(time)
        sin_span = torch.sin(torch.pi * m_span * time.unsqueeze(-1))
        sin_span = torch.cat([sin_span]*6, dim=1)[:, :, None, None]
        cos_span = torch.pi * m_span * torch.cos(torch.pi * m_span * time.unsqueeze(-1))
        cos_span = torch.cat([cos_span]*6, dim=1)[:, :, None, None]
        
        return sin_span, cos_span

    def get_alpha_tildas(self, net_alpha_out, time, sin_span):
        # values
        channel, height, width = net_alpha_out.shape[1], net_alpha_out.shape[2], net_alpha_out.shape[3]

        # matching dimension
        if self.spatial_dimension == '[B, C, H, W]':
            net_alpha_out = net_alpha_out
        
        elif self.spatial_dimension == '[B, C, H, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, -1, width)
        elif self.spatial_dimension == '[B, C, 1, W]':
            net_alpha_out = net_alpha_out.mean(dim=[2], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, height, -1)
        elif self.spatial_dimension == '[B, 1, H, W]':
            net_alpha_out = net_alpha_out.mean(dim=[1], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, -1, -1)

        elif self.spatial_dimension == '[B, C, 1, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[2, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, height, width)
        elif self.spatial_dimension == '[B, 1, H, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, -1, width)
        elif self.spatial_dimension == '[B, 1, 1, W]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 2], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, height, -1)

        elif self.spatial_dimension == '[B, 1, 1, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 2, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, height, width)

        # calculate temporary values
        temp_0 = (net_alpha_out * sin_span) ** 2
        temp_0 = torch.stack([
            torch.sum(temp_0[:, self.M * i:self.M * (i+1), :, :], dim=1)
            for i in range(6)
        ], dim=1)
        
        # calculate tildas
        alpha_0_tilda = 1 - time[:, None, None, None] + temp_0[:, :3, :, :]
        alpha_1_tilda = time[:, None, None, None] + temp_0[:, 3:, :, :]

        return alpha_0_tilda, alpha_1_tilda
    
    def get_alpha_tilda_derivatives(self, net_alpha_out, sin_span, cos_span):
        # values
        channel, height, width = net_alpha_out.shape[1], net_alpha_out.shape[2], net_alpha_out.shape[3]

        # matching dimension
        if self.spatial_dimension == '[B, C, H, W]':
            net_alpha_out = net_alpha_out
        
        elif self.spatial_dimension == '[B, C, H, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, -1, width)
        elif self.spatial_dimension == '[B, C, 1, W]':
            net_alpha_out = net_alpha_out.mean(dim=[2], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, height, -1)
        elif self.spatial_dimension == '[B, 1, H, W]':
            net_alpha_out = net_alpha_out.mean(dim=[1], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, -1, -1)

        elif self.spatial_dimension == '[B, C, 1, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[2, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, -1, height, width)
        elif self.spatial_dimension == '[B, 1, H, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, -1, width)
        elif self.spatial_dimension == '[B, 1, 1, W]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 2], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, height, -1)

        elif self.spatial_dimension == '[B, 1, 1, 1]':
            net_alpha_out = net_alpha_out.mean(dim=[1, 2, 3], keepdim=True)
            net_alpha_out = net_alpha_out.expand(-1, channel, height, width)

        # calculate temporary values
        temp_0 = net_alpha_out * sin_span
        temp_1 = net_alpha_out * cos_span
        
        temp_0 = torch.stack([
            torch.sum(temp_0[:, self.M * i:self.M * (i+1), :, :], dim=1)
            for i in range(6)
        ], dim=1)
        temp_1 = torch.stack([
            torch.sum(temp_1[:, self.M * i:self.M * (i+1), :, :], dim=1)
            for i in range(6)
        ], dim=1)

        temp_2 = temp_0 * temp_1

        # calculate derivative of tildas
        alpha_0_tilda_derivative = -1 + 2 * temp_2[:, :3, :, :]
        alpha_1_tilda_derivative = 1 + 2 * temp_2[:, 3:, :, :]

        return alpha_0_tilda_derivative, alpha_1_tilda_derivative
    
    def get_alpha_hats(self, alpha_0_tilda, alpha_1_tilda):
        temp = alpha_0_tilda + alpha_1_tilda
        alpha_0_hat = alpha_0_tilda / temp
        alpha_1_hat = alpha_1_tilda / temp
        return alpha_0_hat, alpha_1_hat
    
    def get_alpha_hat_derivatives(
        self, alpha_0_tilda, alpha_1_tilda, 
        alpha_0_tilda_derivative, alpha_1_tilda_derivative
        ):
        # calculate derivative of alphas
        alpha_0_hat_derivative = (alpha_0_tilda_derivative * alpha_1_tilda - alpha_0_tilda * alpha_1_tilda_derivative)
        alpha_0_hat_derivative = alpha_0_hat_derivative / ((alpha_0_tilda + alpha_1_tilda) ** 2)
        alpha_1_hat_derivative = - alpha_0_hat_derivative
        return alpha_0_hat_derivative, alpha_1_hat_derivative
    
    def euler_integration(self, x0):
        # initialize xt
        xt = x0

        # get net_alpha outputs
        net_alpha_out = self.net_alpha(x=x0)

        for n in range(self.integration_steps):
            time = torch.ones(x0.shape[0]).type_as(x0) * self.time_span[n].item()
            time_future = torch.ones(x0.shape[0]).type_as(x0) * self.time_span[n + 1].item()

            # get trigonometric spans
            sin_span, cos_span = self.get_trigonometric_spans(time)

            # get alpha tildas and derivatives
            alpha_0_tilda, alpha_1_tilda = self.get_alpha_tildas(net_alpha_out, time, sin_span)
            alpha_0_tilda_future, alpha_1_tilda_future = self.get_alpha_tildas(net_alpha_out, time_future, sin_span)
            alpha_0_tilda_derivative, alpha_1_tilda_derivative = self.get_alpha_tilda_derivatives(
                net_alpha_out, sin_span, cos_span)

            # get alpha hats
            _, alpha_1 = self.get_alpha_hats(alpha_0_tilda, alpha_1_tilda)
            _, alpha_1_future = self.get_alpha_hats(alpha_0_tilda_future, alpha_1_tilda_future)
            delta_alpha = alpha_1_future - alpha_1

            # get derivatives of alpha hats
            alpha_0_derivative, alpha_1_derivative = self.get_alpha_hat_derivatives(
                alpha_0_tilda, alpha_1_tilda, alpha_0_tilda_derivative, alpha_1_tilda_derivative
                )

            # get net_g outputs
            net_g_out = self.net_g(x=xt, interpolants=alpha_1)

            # caculate bt with g and alphas
            bt = alpha_0_derivative * net_g_out[:, :3, : ,:] + alpha_1_derivative * net_g_out[:, 3:, :, :]
            
            # calculate next xt
            xt = xt + bt * delta_alpha
        
        return xt
    
    def forward(self, x0, x1):
        # zero gradient
        self.discriminator.zero_grad()
        self.net_g.zero_grad()
        self.net_alpha.zero_grad()

        # Update Generator: maximize log(D(G(z)))
        pred_x1 = self.euler_integration(x0)
        label = torch.full((x1.shape[0], 1), 1., dtype=torch.float).type_as(x1)  # fake labels are real for generator cost
        output = self.discriminator(pred_x1)
        errG = self.bceloss(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizer_net_alpha.step()

        # zero gradient
        self.discriminator.zero_grad()
        self.net_g.zero_grad()
        self.net_alpha.zero_grad()

        # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        label.fill_(1.)
        output = self.discriminator(x1)
        errD_real = self.bceloss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        label.fill_(0.)
        output = self.discriminator(pred_x1.detach())
        errD_fake = self.bceloss(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        self.optimizer_discriminator.step()

        return errD.item(), errG.item(), D_x, D_G_z1, D_G_z2