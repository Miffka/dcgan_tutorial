from __future__ import print_function
import argparse
import os
import os.path as osp
import random
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a DCGAN')

    parser.add_argument()




    random.seed(args.random_state)
    os.environ["PYTHONHASHSEED"] = str(args.random_state)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False