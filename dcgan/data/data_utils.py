import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from dcgan.config import model_config


def create_dataloader(args):
    dataset = datasets.ImageFolder(
        root=args.dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Check dataloader")

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
    parser.add_argument("--out_dir", default=osp.join(model_config.data_dir, "images"))

    args = parser.parse_args()

    dataloader = create_dataloader(args)
    real_batch = next(iter(dataloader))
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0].to(model_config.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)
        )
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Save training images to {args.out_dir}")
    plt.savefig(osp.join(args.out_dir, "training_images.png"), bbox_inches="tight", pad_inches=0)
