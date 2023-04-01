'''
Train DSNO for unconditional generation on CIFAR-10
'''
import os
import random
from argparse import ArgumentParser
import copy
from functools import partial
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, ChainedScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax import serialization

from models.tddpmm import TDDPMm, get_logsnr_schedule

from models.utils import save_ckpt, load_from_jax_ckpt
from utils.data_helper import data_sampler, sample_data
from utils.dataset import LMDBData
from utils.distributed import setup, cleanup, reduce_loss_dict
from utils.helper import count_params, prepare_pretrain_file
from utils.loss import weightedL1, weightedL2, percepLoss

import lpips


try:
    import wandb
except ImportError:
    wandb = None


def get_idx(t_idx, t_dim=17, num_steps=512, time_step='uniform'):
    if time_step == 'uniform':
        step = num_steps // (t_dim - 1)
        idxs = [step * i for i in t_idx]
    else:
        idxs = [num_steps - 2 * i * i for i in reversed(t_idx)]
    return idxs


def train(model, model_ema,
          dataloader,
          optimizer, scheduler,
          device, config, args,
          rank=0):
    # get configuration
    logsnr_min = config.model.logsnr_min
    logsnr_max = config.model.logsnr_max
    loss_weight = config.model.loss_weight

    grad_clip = config.optim.grad_clip
    t_dim = config.data.t_dim
    t_idx = config.data.t_idx
    num_steps = config.data.num_steps

    ema_decay = config.model.ema_rate
    start_iter = config.training.start_iter

    target_num_t = config.model.num_t # number of time steps to predict
    num_pad = config.model.num_pad    # number of steps for padding (Fourier continuation)
    total_num_t = target_num_t + num_pad    # number of total time steps

    logname = config.log.logname
    save_step = config.eval.save_step
    use_wandb = config.use_wandb if 'use_wandb' in config else False
    # setup wandb
    if use_wandb and wandb:
        run = wandb.init(entity=config.log.entity,
                         project=config.log.project,
                         group=config.log.group,
                         config=config.to_container(),
                         reinit=True,
                         settings=wandb.Settings(start_method='fork'))

    # prepare exp dir
    base_dir = f'exp/{logname}'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # prepare time input
    logsnr_fn = get_logsnr_schedule(logsnr_max, logsnr_min)

    t0, t1 = 1., config.data.epsilon
    timesteps = torch.linspace(t0, t1, num_steps + 1, device=device)
    idxs = get_idx(t_idx=t_idx, t_dim=t_dim, num_steps=num_steps, 
                   time_step=config.data.time_step)

    timesteps = timesteps[idxs[-total_num_t:]]
    logsnr = logsnr_fn(timesteps)
    # define loss function
    loss_type = config.training.loss
    if loss_type == 'L1':
        criterion = weightedL1
    elif loss_type == 'L2':
        criterion = weightedL2
    else:
        # perceptual loss, VGG-based versions works the best. 
        loss_fn = lpips.LPIPS(net=loss_type, lpips=config.training.lpips).cuda()
        criterion = partial(percepLoss, loss_fn=loss_fn)

    # compute loss weighting
    if loss_weight == 'snr':
        weights = torch.sqrt(torch.exp(logsnr)).clamp(1.0, 10000.0)
    else:
        weights = torch.ones_like(logsnr, device=device)

    # training
    if rank == 0:
        pbar = tqdm(range(start_iter, config.training.n_iters), dynamic_ncols=True)
    else:
        pbar = range(start_iter, config.training.n_iters)
    log_dict = {}
    dataloader = sample_data(dataloader)

    for e in pbar:
        model.train()
        states = next(dataloader)
        # B, T, C, H, W
        states = states.to(device)

        in_state = states[:, 0]                     # (B, C, H, W) Gaussian noise
        target_state = states[:, -target_num_t:]    # (B, T-1, C, H, W) target trajectory

        pred = model(in_state, logsnr)              # (B, T-1, C, H, W) output trajectory
        target_pred = pred[:, -target_num_t:]       # (B, T-1, C, H, W) predicted trajectory. This line does not change anything, just for future flexibility in case we want to add Fourier continuation.

        loss, loss_ts  = criterion(target_pred, target_state, weights)
        # update model
        model.zero_grad()

        loss.backward()
        if grad_clip > 0.0:     # gradient clipping is not necessary for training dsno, but we keep it here as an additional option. 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        log_dict['train_loss'] = loss
        log_dict['loss_ts'] = loss_ts
        scheduler.step()

        reduced_log_dict = reduce_loss_dict(log_dict)
        train_loss = reduced_log_dict['train_loss'].item()
        if rank == 0:
            pbar.set_description(
                (
                    f'Epoch :{e}, Loss: {train_loss}'
                )
            )
        log_state = {f'loss at time {timesteps[-target_num_t:][i]}': loss_ts[i].item() for i in range(target_num_t)}
        log_state['train MSE'] = train_loss

        # Update moving average of the model parameters
        with torch.no_grad():
            for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_decay))
            for b_ema, b in zip(model_ema.buffers(), model.buffers()):
                b_ema.copy_(b)

        if e % save_step == 0 and e > 0:
            if rank == 0:
                save_path = os.path.join(save_ckpt_dir,
                                         f'solver-model_{e}.pt')
                save_ckpt(save_path,
                          model=model, model_ema=model_ema,
                          optim=optimizer, scheduler=scheduler,
                          args=args)
            
            # generate 50k images to save_img_dir for evaluation
            # torch.distributed.barrier()
            # if config.eval.test_fid and e > 0 and rank == 0:
            #     fid_score = compute_fid(model_ema, temb=logsnr)
            #     log_state['FID'] = fid_score
        if use_wandb and wandb:
            wandb.log(log_state)

    if rank == 0:
        save_path = os.path.join(save_ckpt_dir, 'solver-model_final.pt')
        save_ckpt(save_path,
                  model=model, model_ema=model_ema,
                  optim=optimizer, args=args)

    if use_wandb and wandb:
        run.finish()


def run(train_loader,
        config, args, device, rank=0):
    # create model
    model = TDDPMm(config).to(device)
    if 'init_ckpt' in config.training:  
        # initialize UNet part from pre-trained model
        ckpt_path = config.training.init_ckpt
        prepare_pretrain_file(ckpt_path)
        with open(ckpt_path, 'rb') as f:
            ckpt = serialization.from_bytes(target=None, encoded_bytes=f.read())['ema_params']
            load_from_jax_ckpt(model, ckpt, config)

    model_ema = copy.deepcopy(model)
    model_ema.eval()
    model_ema.requires_grad_(False)

    num_params = count_params(model)
    print(f'number of parameters: {num_params}')
    config['num_params'] = num_params

    # define optimizer and criterion
    if config.optim.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=config.optim.lr, 
                         betas=(config.optim.beta1, config.optim.beta2))
    elif config.optim.optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=config.optim.lr)
    scheduler1 = LinearLR(optimizer, start_factor=0.001, total_iters=config.optim.warmup)
    scheduler2 = MultiStepLR(optimizer,
                             milestones=config.optim.milestone,
                             gamma=0.5)
    scheduler = ChainedScheduler([scheduler1, scheduler2])

    # Load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        model_ema.load_state_dict(ckpt['ema'])
        print(f'Load weights from {args.ckpt}')
        optimizer.load_state_dict(ckpt['optim'])
        print(f'Load optimizer state..')
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f'Load scheduler state..')
        config.training.start_iter = scheduler._schedulers[0].last_epoch

    if args.distributed:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    train(model, model_ema, train_loader,
          optimizer, scheduler,
          device, config, args,
          rank=rank)


def subprocess_fn(rank, args):
    # setup
    if args.distributed:
        setup(rank, args.num_gpus, port=f'{args.port}')
    print(f'Running on rank: {rank}')

    config = OmegaConf.load(args.config)
    # parse configuration file
    torch.cuda.set_device(rank)
    device = torch.device('cuda')
    batchsize = config.training.batchsize
    if config.data.num_sample == -1:
        # use all data in the database
        num_data = None
    else:
        num_data = config.data.num_sample
        
    if args.log and rank == 0:
        config.use_wandb = True
    else:
        config.use_wandb = False

    config.seed = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    db_path = config.data.datapath
    trainset = LMDBData(db_path, 
                        data_shape=config.data.shape,
                        dims=config.data.dims,  
                        t_idx=config.data.t_idx, 
                        num_data=num_data)

    train_loader = DataLoader(trainset, batch_size=batchsize,
                              sampler=data_sampler(trainset,
                                                   shuffle=True,
                                                   distributed=args.distributed),
                              num_workers=4,
                              persistent_workers=True,
                              pin_memory=True,
                              drop_last=True)

    run(train_loader, config=config, args=args, device=device, rank=rank)

    if args.distributed:
        cleanup()
    print(f'Process {device} exits...')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/cifar/tunet.yaml', help='configuration file')
    parser.add_argument('--ckpt', type=str, default=None, help='Which checkpoint to initialize the model')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--port', type=str, default='9037')
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    if args.seed is None:
        args.seed = random.randint(0, 100000)
    if args.distributed:
        mp.spawn(subprocess_fn, args=(args,), nprocs=args.num_gpus)
    else:
        subprocess_fn(0, args)