# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
from absl import flags
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples_ai, infiniteloop

from adaptive_multidimensional_path import AdaptiveMultidimensionalPath
from torchcfm.models.unet.unet_modified import UNetModelWrapper

from unet.unet_model import UNet

from discriminator import Discriminator

from config_ai import config

# Fixed parameters
######################################################################################
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", f"{config['saving_path']}", help="output_directory")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", config['lr'], help="target learning rate")
flags.DEFINE_float("lr_disc", config['lr_disc'], help="target learning rate for discriminator")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", config['total_steps'], help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", config['warmup_steps'], help="learning rate warmup")
flags.DEFINE_integer("warmup_disc", config['warmup_steps_disc'], help="learning rate warmup for discriminator")
flags.DEFINE_integer("batch_size", config['batch_size'], help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    5000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#######################################################################################


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def warmup_lr_disc(step):
    return min(step, FLAGS.warmup_disc) / FLAGS.warmup_disc

def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
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
    ).to('cuda:0')  # new dropout + bs of 128

    checkpoint = torch.load(f"./models/{config['date_net_g']}/model_cifar10_weights_step_400000.pt")
    state_dict = checkpoint['ema_model']
    net_g.load_state_dict(state_dict)

    # disable randomness, dropout, etc...
    net_g.eval()

    model = UNet(n_channels=3, n_classes=int(config['M'] * 6)).to('cuda:0')
    discriminator = Discriminator(ndf=config['ndf'])

    optimizer_net_alpha = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=FLAGS.lr_disc)

    adaptivemultidimensionalpath = AdaptiveMultidimensionalPath(
        net_g=net_g,
        net_alpha=model,
        discriminator=discriminator,
        optimizer_net_alpha=optimizer_net_alpha,
        optimizer_discriminator=optimizer_discriminator,
        integration_steps=config['integration_steps'],
        spatial_dimension=config['spatial_dimension']
        ).to('cuda:0')

    ema_model = copy.deepcopy(model)
    sched_net_alpha = torch.optim.lr_scheduler.LambdaLR(optimizer_net_alpha, lr_lambda=warmup_lr)
    sched_discriminator = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminator, lr_lambda=warmup_lr_disc)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # Trainer   
    savedir = FLAGS.output_dir + config['date'] + "/"
    os.makedirs(savedir, exist_ok=True)

    # Train
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1).to(device)
            Loss_D, Loss_G, D_x, D_G_z1, D_G_z2 = adaptivemultidimensionalpath(x0, x1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)  # new
            sched_net_alpha.step()
            sched_discriminator.step()
            ema(model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            with torch.no_grad():
                if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                    # samples
                    generate_samples_ai(adaptivemultidimensionalpath, FLAGS.parallel, savedir, step, config['integration_steps'], net_="normal")
                    generate_samples_ai(adaptivemultidimensionalpath, FLAGS.parallel, savedir, step, config['integration_steps'], net_="normal")

                    # models
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched_net_alpha": sched_net_alpha.state_dict(),
                            "sched_discriminator": sched_discriminator.state_dict(),
                            "optimizer_net_alpha": optimizer_net_alpha.state_dict(),
                            "optimizer_discriminator": optimizer_discriminator.state_dict(),
                            "step": step,
                        },
                        savedir + f"model_cifar10_weights_step_{step}.pt",
                    )