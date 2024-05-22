import os
import gc  # Garbage Collector
import random
import numpy as np
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from torchcfm.models.unet.unet_modified import UNetModelWrapper

from config_ai import config

from solver import SimplePathEulerSolver



def main():
    # epoch and date list
    date = '240320_3'
    model_step = 400000
    N = 200

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
    batch_size = 1000
    num_iteration = 50
    sample_saving_path = f'./sample/{date}/{N}/{model_step}'
    os.makedirs(sample_saving_path, exist_ok=True)

    # load model
    model = UNetModelWrapper(
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

    checkpoint = torch.load(f'./models/{date}/model_cifar10_weights_step_{model_step}.pt')
    state_dict = checkpoint['ema_model']
    model.load_state_dict(state_dict)

    # disable randomness, dropout, etc...
    model.eval()

    # solver
    solver = SimplePathEulerSolver(model, N=N)

    # test
    with torch.no_grad():
        for step in tqdm(range(num_iteration)):

            # make x0
            x_start = torch.randn(batch_size, 3, 32, 32)
            x_start = x_start.to('cuda:0')

            # prediction
            pred_xt = solver(x_start)
            pred_xt = pred_xt.clip(-1, 1)
            pred_xt = pred_xt / 2 + 0.5

            # Save images
            for img_idx in range(batch_size):
                save_image(pred_xt[img_idx], "{}/{}.png".format(sample_saving_path, img_idx + step * batch_size))

    # dealloc
    model.to('cpu')
    del model, solver  # Delete model and solver to free memory
    torch.cuda.empty_cache()  # Clear unused memory
    gc.collect()  # Explicitly call garbage collector

if __name__ == '__main__':
    main()