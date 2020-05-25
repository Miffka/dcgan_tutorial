from __future__ import print_function

import argparse
import os
import os.path as osp
import random

import numpy as np
import torch

from dcgan.config import model_config
from dcgan.data.data_utils import create_dataloader
from dcgan.models.dcgan import create_nets

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DCGAN")

    parser.add_argument("--random_state", type=int, default=24, help="Random state")

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

    ### Fix random seeds ###
    random.seed(args.random_state)
    os.environ["PYTHONHASHSEED"] = str(args.random_state)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ### Create dataloader ###
    dataloader = create_dataloader(args)

    ### Create nets ###
    netG, netD = create_nets(args)
