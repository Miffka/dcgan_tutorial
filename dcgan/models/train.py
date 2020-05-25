import argparse
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dcgan.config import model_config
from dcgan.data.data_utils import create_dataloader
from dcgan.models.net import create_nets
from dcgan.models.train_utils import train_gan

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DCGAN")

    ### Main parameters ###
    parser.add_argument("--task_name", default="sample", help="Experiment name")
    parser.add_argument("--random_state", type=int, default=24, help="Random state")
    parser.add_argument("--verbose", action="store_true", help="Whether to print losses during training")
    parser.add_argument("--logdir", default=model_config.log_dir, help="Folder with tensorboard runs")
    parser.add_argument("--force_cpu", action="store_true", help="Whether to train on CPU")

    ### Data parameters ###
    parser.add_argument(
        "--dataroot", default=osp.join(model_config.data_dir, "celeba"), help="Root directory for dataset"
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Spatial size of training images. All images will be resized to this size using a transformer.",
    )
    parser.add_argument(
        "--test_mode", action="store_true", help="Whether to take only a small batch of data (debug mode)"
    )
    parser.add_argument("--test_size", type=int, default=512, help="Size of dataset in debug mode")

    ### Network parameters ###
    parser.add_argument(
        "--nc", type=int, default=3, help="Number of channels in the training images. For color images this is 3"
    )
    parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")

    ### Training parameters ###
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Number of workers for dataloader")
    parser.add_argument("--beta1", type=float, default=0.5, help="Number of workers for dataloader")

    args = parser.parse_args()

    ### Configure logging ###
    ts = time.strftime("%Y_%m_%d__%H_%M_%S")
    run_folder = osp.join(args.logdir, args.task_name, ts)
    save_folder = osp.join(model_config.model_dir, args.task_name)
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    writer = SummaryWriter(run_folder)

    ### Fix random seeds ###
    random.seed(args.random_state)
    os.environ["PYTHONHASHSEED"] = str(args.random_state)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ### Define device ###
    device = model_config.device if not args.force_cpu else torch.device("cpu")

    ### Create dataloader ###
    dataloader = create_dataloader(args)

    ### Create nets ###
    netG, netD = create_nets(args)

    ### Define criterions and optimizers ###
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    ### Train model ###
    train_gan(
        netD,
        netG,
        dataloader,
        real_label,
        fake_label,
        criterion,
        optimizerD,
        optimizerG,
        fixed_noise,
        args.nz,
        device,
        args.num_epochs,
        writer=writer,
        verbose=False,
        save_folder=save_folder,
    )
