import os
import click
from tqdm import tqdm
import pickle
from PIL import Image


import torch
import torch.multiprocessing as mp

from omegaconf import OmegaConf


from models.tddpmm import TDDPMm, get_logsnr_schedule
from utils.helper import dict2namespace

try:
    import wandb
except ImportError:
    wandb = None


def save2dir(images, outdir, curr):
    num_imgs = images.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(images[j])
        img_path = os.path.join(outdir, f'{j + curr}.png')
        im.save(img_path)    
    return curr + num_imgs


def get_random_label(batchsize, num_class, device):
    labels = torch.randint(low=0, high=num_class, size=(batchsize, ), device=device)
    return labels


def get_idx(t_idx, t_dim=17, num_steps=512, time_step='uniform'):
    if time_step == 'uniform':
        step = num_steps // (t_dim - 1)
        idxs = [step * i for i in t_idx]
    else:
        idxs = [num_steps - 2 * i * i for i in reversed(t_idx)]
    return idxs



@torch.no_grad()
def uncond_generate(model, config, device, outdir, curr=0, batch=50, num_imgs=50000):
    logsnr_min = config['model']['logsnr_min']
    logsnr_max = config['model']['logsnr_max']

    t_dim = config['data']['t_dim']
    t_idx = config['data']['t_idx']
    num_steps = config['data']['num_steps']
    img_size = config['data']['image_size']

    target_num_t = config['model']['num_t'] # number of time steps to predict
    num_pad = config['model']['num_pad']    # number of steps for padding (Fourier continuation)
    total_num_t = target_num_t + num_pad    # number of total time steps

    # prepare time input
    logsnr_fn = get_logsnr_schedule(logsnr_max, logsnr_min)

    t0, t1 = 1., config['data']['epsilon']
    timesteps = torch.linspace(t0, t1, num_steps + 1, device=device)
    idxs = get_idx(t_idx=t_idx, t_dim=t_dim, num_steps=num_steps, 
                   time_step=config['data']['time_step'])

    timesteps = timesteps[idxs[-total_num_t:]]
    logsnr = logsnr_fn(timesteps)
    model.eval()

    num_batches = (num_imgs - 1) // batch + 1
    for i in range(num_batches):
        if i == num_batches - 1 and (num_imgs % batch > 0):
            curr_batchsize = num_imgs % batch
        else:
            curr_batchsize = batch
        init_x = torch.randn((curr_batchsize, 3, img_size, img_size), device=device)
        pred = model(init_x, logsnr)[:, -1]
        imgs = pred.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1) # B, H, W, C
        imgs = imgs.cpu().numpy()
        curr = save2dir(imgs, outdir, curr=curr)
    print(f'{num_imgs} images generated to {outdir}')


@torch.no_grad()
def generate(model, config, num_imgs, batch, device, num_gpus):
    # get configuration
    logsnr_min = config['model']['logsnr_min']
    logsnr_max = config['model']['logsnr_max']

    t_dim = config['data']['t_dim']
    t_idx = config['data']['t_idx']
    num_steps = config['data']['num_steps']

    target_num_t = config['model']['num_t'] # number of time steps to predict
    num_pad = config['model']['num_pad']    # number of steps for padding (Fourier continuation)
    total_num_t = target_num_t + num_pad    # number of total time steps

    logname = config['log']['logname']

    # prepare exp dir
    base_dir = f'exp/{logname}'
    save_img_dir = f'{base_dir}/samples'
    os.makedirs(save_img_dir, exist_ok=True)
    x_dir = os.path.join(base_dir, 'random-init')
    os.makedirs(x_dir, exist_ok=True)

    # prepare time input
    logsnr_fn = get_logsnr_schedule(logsnr_max, logsnr_min)

    t0, t1 = 1., config['data']['epsilon']
    timesteps = torch.linspace(t0, t1, num_steps + 1, device=device)
    idxs = get_idx(t_idx=t_idx, t_dim=t_dim, num_steps=num_steps, 
                   time_step=config['data']['time_step'])

    timesteps = timesteps[idxs[-total_num_t:]]
    logsnr = logsnr_fn(timesteps)

    batchsize = batch
    num_iter = num_imgs // batchsize
    if num_gpus:
        logsnr = logsnr.repeat([num_gpus, 1])
    model.eval()
    curr = 0
    for e in tqdm(range(num_iter)):
        # B, T, C, H, W
        y = get_random_label(batchsize, num_class=1000, device=device)
        y = y.long()
        init_state = torch.randn((batchsize, 3, 64, 64), device=device)

        pred = model(init_state, logsnr, y)
        imgs = pred[:, -1]
        
        imgs = imgs.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)   #B, C, H, W -> B, H, W, C
        img_arr = imgs.cpu().numpy()
        curr = save2dir(img_arr, save_img_dir, curr=curr)
    print(f'{num_imgs} images generated to {save_img_dir}')


@torch.no_grad()
def subproc_generate(rank, config, ckpt, num_imgs, seed, batch, num_gpus=1):
    device = torch.device('cuda')
    config['seed'] = seed
    
    # set random seed
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    model = TDDPMm(config).to(device)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict['ema'])
    print(f'Load weights from {ckpt}')
    generate(model, config, num_imgs, batch, device, num_gpus)


#----------------------------------------------------------------------------
@click.command()
@click.option('--config', 'config_path',    help='Path to the configuration file', metavar='PATH',      type=str, required=True)
@click.option('--ckpt',                     help='Path to the checkpoint, if directory, evaluate all checkpoints under thid directory', metavar='PATH',          type=str, default='all')
@click.option('--num', 'num_imgs',          help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--num_gpus',             help='Number of gpus', metavar='INT',                    type=click.IntRange(min=1), default=8, show_default=True)


def main(config_path, ckpt, num_imgs, seed, batch, num_gpus):
    torch.backends.cudnn.benchmark = True
    config = OmegaConf.load(config_path)
    if num_gpus > 1:
        mp.spawn(subproc_generate, args=(config, ckpt, num_imgs, seed, batch, num_gpus), nprocs=num_gpus)
    else:
        subproc_generate(0, config, ckpt, num_imgs, seed, batch, num_gpus)


if __name__ == '__main__':
    main()