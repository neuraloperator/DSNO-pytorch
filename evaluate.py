import os
from omegaconf import OmegaConf
from argparse import ArgumentParser
import wandb
import numpy as np

import torch
import torch.multiprocessing as mp

from models.tddpmm import TDDPMm

from generate_imagenet import uncond_generate



def main(args):
    config = OmegaConf.load(args.config)
    device = torch.device('cuda')
    if args.log:
        config.use_wandb = True
    else:
        config.use_wandb = False
    
    config.seed = args.seed
    torch.manual_seed(args.seed)

    if args.log:
        run = wandb.init(entity=config['log']['entity'],
                         project=config['log']['project'],
                         group=config['log']['group'],
                         config=config,
                         reinit=True,
                         settings=wandb.Settings(start_method='fork'))    

    model = TDDPMm(config).to(device)
    model.requires_grad_(False)
    ckpt_dir = os.path.join('exp', config.log.logname, 'ckpts')
    outdir = os.path.join('exp', config.log.logname, 'images')
    os.makedirs(outdir, exist_ok=True)
    ref_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'


    for i in range(args.start, args.end + args.step, args.step):
        ckpt_path = args.ckpt if args.ckpt else os.path.join(ckpt_dir, f'solver-model_{i}.pt')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['ema'])
        print(f'generate images from checkpoint {i}.pt')
        uncond_generate(model, config, device, outdir, batch=args.batchsize, num_imgs=args.num_imgs)
    print('Done!')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/cifar/tddpmm_t4-quad-snr-256-vgg-radam.yaml', help='configuration file')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to initialize the model')
    parser.add_argument('--batchsize', type=int, help='batchsize', default=100)
    parser.add_argument('--num_imgs', type=int, default=50000)
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    parser.add_argument('--seed', type=int, default=321)

    args = parser.parse_args()
    main(args)