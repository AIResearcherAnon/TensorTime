import copy

import torch
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

# from sampler import Sampler

from solver import SimplePathEulerSolver, AdaptivePathEulerSolver

from config import config

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_samples(model, parallel, savedir, step, t_span=torch.linspace(0, 1, 101, device=device), net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    sampler = SimplePathEulerSolver(model_, N=100)
    with torch.no_grad():
        pred_x1 = sampler(
            x0=torch.randn(64, 3, 32, 32, device=device)
        )
        pred_x1 = pred_x1.clip(-1, 1)
        pred_x1 = pred_x1 / 2 + 0.5
    save_image(pred_x1, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def generate_samples_ai(model, parallel, savedir, step, integration_steps, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    t_span=torch.linspace(0, 1, integration_steps+1, device=device)
    
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    sampler = AdaptivePathEulerSolver(model_, N=100)
    with torch.no_grad():
        pred_x1 = sampler(
            x0=torch.randn(64, 3, 32, 32, device=device)
        )
        pred_x1 = pred_x1.clip(-1, 1)
        pred_x1 = pred_x1 / 2 + 0.5
    save_image(pred_x1, savedir + f"{net_}_generated_FM_images_step_{step}_ai.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
