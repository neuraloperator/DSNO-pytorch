'''
Train DSNO for conditional generation
'''
import os
import random
from argparse import ArgumentParser
import copy
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, ChainedScheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from flax import serialization

from models.tddpmm import TDDPMm, get_logsnr_schedule

from models.utils import save_ckpt, load_from_jax_ckpt
from utils.data_helper import data_sampler, sample_data
from utils.dataset import ImageNet
from utils.distributed import setup, cleanup, reduce_loss_dict
from utils.helper import count_params, get_latest_ckpt, prepare_pretrain_file
from utils.loss import weightedL1, weightedL2, percepLoss

import lpips

try:
    import apex
except ImportError:
    apex = None

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
          criterion,
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
    accum_grad_iter = config.training.accum_grad_iter

    target_num_t = config.model.num_t # number of time steps to predict
    num_pad = config.model.num_pad    # number of steps for padding (Fourier continuation)
    total_num_t = target_num_t + num_pad    # number of total time steps

    logname = config.log.logname
    save_step = config.eval.save_step
    use_wandb = config.use_wandb if 'use_wandb' in config else False

    enable_amp = args.amp
    # setup wandb
    if use_wandb and wandb:
        run = wandb.init(entity=config.log.entity,
                         project=config.log.project,
                         group=config.log.group,
                         config=config,
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
    timesteps = torch.linspace(t0, t1, num_steps + 1)
    idxs = get_idx(t_idx=t_idx, t_dim=t_dim, num_steps=num_steps, 
                   time_step=config.data.time_step)

    timesteps = timesteps[idxs[-total_num_t:]]
    logsnr = logsnr_fn(timesteps).to(device)
    # compute loss weighting
    if loss_weight == 'snr':
        weights = torch.sqrt(torch.exp(logsnr)).clamp(1.0, 10000.0)
    else:
        weights = 1.0
    # training
    if rank == 0:
        pbar = tqdm(range(config.training.n_iters), dynamic_ncols=True)
    else:
        pbar = range(config.training.n_iters)
    log_dict = {}
    dataloader = sample_data(dataloader)
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    for e in pbar:
        model.train()
        states, y = next(dataloader)
        # B, T, C, H, W
        states = states.to(device)
        y = y.to(device).long()
        in_state = states[:, 0]
        target_state = states[:, -target_num_t:]

        with torch.autocast(device_type='cuda', enabled=enable_amp):
            pred = model(in_state, logsnr, y)
            target_pred = pred[:, -target_num_t:]
            loss, loss_ts  = criterion(target_pred, target_state, weights)
            loss = loss / accum_grad_iter
        # accumulate gradient
        scaler.scale(loss).backward()

        if (e + 1) % accum_grad_iter == 0:
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
            scheduler.step()
            # Update moving average of the model parameters
            with torch.no_grad():
                for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_decay))
                for b_ema, b in zip(model_ema.buffers(), model.buffers()):
                    b_ema.copy_(b)

        log_dict['train_loss'] = loss
        log_dict['loss_ts'] = loss_ts
        

        reduced_log_dict = reduce_loss_dict(log_dict)
        train_loss = reduced_log_dict['train_loss'].item()
        if rank == 0:
            pbar.set_description(
                (
                    f'Epoch :{e}, Loss: {train_loss}'
                )
            )
            log_state = {f'loss at time {timesteps[i].item()}': loss_ts[i].item() for i in range(target_num_t)}
            log_state['train MSE'] = train_loss

        check_step = save_step * accum_grad_iter
        if (e + 1) % check_step == 0 and e > accum_grad_iter:
            # memory_use = psutil.Process().memory_info().rss / (1024 * 1024)
            # print(f'Step {e}; Memory usage: {memory_use} MB')
            if rank == 0:
                iter_idx = scheduler._schedulers[0].last_epoch
                save_image(pred[:, -1] * 0.5 + 0.5,
                           f'{save_img_dir}/pred_{iter_idx}.png',
                           nrow=8)
                save_image(states[:, -1] * 0.5 + 0.5,
                           f'{save_img_dir}/truth_{iter_idx}.png',
                           nrow=8)
                save_path = os.path.join(save_ckpt_dir,
                                         f'solver-model_{iter_idx}.pt')
                save_ckpt(save_path,
                          model=model, model_ema=model_ema,
                          optim=optimizer, scheduler=scheduler,
                          args=args)

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
    config.num_params = num_params

    # define optimizer and criterion
    if args.amp and apex is not None:
        optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=config.optim.lr, adam_w_mode=False)
    else:
        optimizer = Adam(model.parameters(), lr=config.optim.lr)
    scheduler1 = LinearLR(optimizer, start_factor=0.001, total_iters=config.optim.warmup)
    scheduler2 = MultiStepLR(optimizer,
                             milestones=config.optim.milestone,
                             gamma=0.5)
    scheduler = ChainedScheduler([scheduler1, scheduler2])

    # Load checkpoint 
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        logname = config.log.logname
        base_dir = f'exp/{logname}'
        save_ckpt_dir = os.path.join(base_dir, 'ckpts')
        os.makedirs(save_ckpt_dir, exist_ok=True)
        ckpt_path = get_latest_ckpt(save_ckpt_dir)
        
    if ckpt_path != -1:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model_ema.load_state_dict(ckpt['ema'])
        print(f'Load weights from {ckpt_path}')
        optimizer.load_state_dict(ckpt['optim'])
        print(f'Load optimizer state..')
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f'Load scheduler state..')
        config.training.start_iter = scheduler._schedulers[0].last_epoch

    if args.distributed:
        model = DDP(model, device_ids=[device], broadcast_buffers=False)
    if config.training.loss == 'L1':
        criterion = weightedL1
    else:
        criterion = weightedL2
    train(model, model_ema, train_loader,
          criterion,
          optimizer, scheduler,
          device, config, args,
          rank=rank)


def subprocess_fn(rank, args):
    # setup
    torch.cuda.set_device(args.local_rank)
    if args.distributed:
        setup(rank, args.world_size, master_addr=args.master_addr,port=args.port)
    print(f'Running on rank: {rank}')
    
    config = OmegaConf.load(args.config)
    # parse configuration file
    
    device = torch.device('cuda')
    batchsize = config.training.batchsize
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
    trainset = ImageNet(db_path, 
                        data_shape=config.data.shape,
                        dims=config.data.dims,  
                        t_idx=config.data.t_idx)

    train_loader = DataLoader(trainset, batch_size=batchsize,
                              sampler=data_sampler(trainset,
                                                   shuffle=True,
                                                   distributed=args.distributed),
                              pin_memory=True,
                              num_workers=4,
                              persistent_workers=True,
                              drop_last=True)

    run(train_loader, config=config, args=args, device=device, rank=rank)
    if args.distributed:
        cleanup()
    print(f'Process on {device} exits...')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/cifar/tunet.yaml', help='configuration file')
    parser.add_argument('--ckpt', type=str, default=None, help='Which checkpoint to initialize the model')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    parser.add_argument('--seed', type=int, default=321)

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_gpus_per_node', type=int, default=1)
    parser.add_argument('--port', type=str, default='9039')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision')
    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_gpus_per_node
    args.distributed = args.world_size > 1
    
    if args.num_gpus_per_node > 1:
        processes = []
        for rank in range(args.num_gpus_per_node):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_gpus_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = mp.Process(target=subprocess_fn, args=(global_rank, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        subprocess_fn(0, args)