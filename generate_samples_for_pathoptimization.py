import os
import gc  # Garbage Collector
import random
import numpy as np
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from torchcfm.models.unet.unet_modified import UNetModelWrapper

from adaptive_multidimensional_path import AdaptiveMultidimensionalPath

from unet.unet_model import UNet

from config import config

from solver import AdaptivePathEulerSolver



def main():
    # epoch and date list
    date_net_g = '240320_4'
    date_net_alpha = '240517_1'
    net_alpha_step = 200000
    spatial_dimension = '[B, C, H, W]'
    M = 10
    N = 10

    # INITIALIZE
    #######################################################
    random_seed = config['seed']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #######################################################

    # configs
    batch_size = 500
    num_iteration = 100
    sample_saving_path = f'./sample/{date_net_alpha}/{N}/{net_alpha_step}'
    os.makedirs(sample_saving_path, exist_ok=True)

    ##############################################################
    # MODEL
    ## load net_g
    net_g = UNetModelWrapper(
        dim=(32, 32),
        in_channels=6,
        out_channels=6,
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to('cuda:0')

    checkpoint = torch.load(f'./models/{date_net_g}/model_cifar10_weights_step_400000.pt')
    state_dict = checkpoint['ema_model']
    net_g.load_state_dict(state_dict)

    # disable randomness, dropout, etc...
    net_g.eval()

    ## load net_alpha
    net_alpha = UNet(n_channels=3, n_classes=int(M * 6)).to('cuda:0')

    checkpoint = torch.load(f'./models/{date_net_alpha}/model_cifar10_weights_step_{net_alpha_step}.pt')
    state_dict = checkpoint['ema_model']
    net_alpha.load_state_dict(state_dict)
    net_alpha.eval()

    ## load adaptive interpolants
    adaptive_interpolants = AdaptiveMultidimensionalPath(
        net_g=net_g,
        net_alpha=net_alpha,
        discriminator=None,
        optimizer_net_alpha=None,
        optimizer_discriminator=None,
        integration_steps=N,
        spatial_dimension=spatial_dimension
    ).to('cuda:0')
    ##############################################################
                
    # solver
    solver = AdaptivePathEulerSolver(adaptive_interpolants, N=N)

    # test
    with torch.no_grad():
        for step in tqdm(range(num_iteration)):

            # make x0
            x_start = torch.randn(batch_size, 3, 32, 32)
            x_start = x_start.to('cuda:0')

            # prediction
            pred_x1 = solver(x_start)
            pred_x1 = pred_x1.clip(-1, 1)
            pred_x1 = pred_x1 / 2 + 0.5

            # Save images
            for img_idx in range(batch_size):
                save_image(pred_x1[img_idx], "{}/{}.png".format(sample_saving_path, img_idx + step * batch_size))

    # dealloc
    adaptive_interpolants.to('cpu')
    del adaptive_interpolants, solver  # Delete model and solver to free memory
    torch.cuda.empty_cache()  # Clear unused memory
    gc.collect()  # Explicitly call garbage collector

if __name__ == '__main__':
    main()