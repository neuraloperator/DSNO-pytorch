import torch
import re
import numpy as np


def save_ckpt(path,
              model,
              model_ema,
              optim=None,
              scheduler=None,
              args=None):
    '''
    saves checkpoint and configurations to dir/name
    :param args: dict of configuration
    :param g_ema: moving average

    :param optim:
    '''
    ckpt_path = path
    if args and args.distributed:
        model_ckpt = model.module
    else:
        model_ckpt = model
    state_dict = {
        'model': model_ckpt.state_dict(),
        'ema': model_ema.state_dict(),
        'args': args
    }

    if optim is not None:
        state_dict['optim'] = optim.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()

    torch.save(state_dict, ckpt_path)
    print(f'checkpoint saved at {ckpt_path}')


def build_state_map(config):

    attn_resolutions = config.model.attn_resolutions
    num_resolution = len(config.model.ch_mult)
    num_resblocks = config.model.num_res_blocks
    all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolution)]
    use_time_conv = config.model.time_conv
    num_classes = config.model.num_classes
    m_idx = 2
    key_map = {
        0: 'dense0', 
        1: 'dense1', 
    }
    if num_classes > 0:
        # add class embedding
        key_map[m_idx] = 'class_emb'
        m_idx += 1
    key_map[m_idx] = 'conv_in'
    m_idx += 1

    # downsample
    for i_level in range(num_resolution):
        for j_block in range(num_resblocks):
            # residual block
            value = f'down_{i_level}.block_{j_block}'
            key_map[m_idx] = value 
            m_idx += 1
            if all_resolutions[i_level] in attn_resolutions:
                # attn block
                value = f'down_{i_level}.attn_{j_block}'
                key_map[m_idx] = value
                m_idx += 1
            if use_time_conv:
                value = f'down_{i_level}.time_conv{j_block}'
                key_map[m_idx] = value
                m_idx += 1
        if i_level != num_resolution - 1:
            # downsample block
            value = f'down_{i_level}.downsample'
            key_map[m_idx] = value
            m_idx += 1
    
    value = 'mid.block_1'
    key_map[m_idx] = value
    
    m_idx += 1
    value = 'mid.attn_1'
    key_map[m_idx] = value 

    m_idx += 1
    value = 'mid.time_conv'
    key_map[m_idx] = value

    m_idx += 1
    value = 'mid.block_2'
    key_map[m_idx] = value
    
    m_idx += 1
    for i_level in reversed(range(num_resolution)):
        for j_block in range(num_resblocks + 1):
            # residual block
            value = f'up_{i_level}.block_{j_block}'
            key_map[m_idx] = value
            m_idx += 1

            if all_resolutions[i_level] in attn_resolutions:
                # attn block
                value = f'up_{i_level}.attn_{j_block}'
                key_map[m_idx] = value
                m_idx += 1
            if use_time_conv:
                value = f'up_{i_level}.time_conv{j_block}'
                key_map[m_idx] = value
                m_idx += 1
        if i_level != 0:
            # upsample block
            value = f'up_{i_level}.upsample'
            key_map[m_idx] = value
            m_idx += 1
    value = 'norm_out'
    key_map[m_idx] = value

    m_idx += 1
    value = 'conv_out'
    key_map[m_idx] = value
    return key_map


def get_key_map(config):
    # attention [16, 8]
    attn_resolutions = config.model.attn_resolutions
    num_resolution = len(config.model.ch_mult)
    num_resblocks = config.model.num_res_blocks
    all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolution)]    
    num_classes = config.model.num_classes

    m_idx = 2
    key_map = {
        0: 'dense0', 
        1: 'dense1', 
    }
    if num_classes > 0:
        # add class embedding
        key_map[m_idx] = 'class_emb'
        m_idx += 1
    key_map[m_idx] = 'conv_in'
    m_idx += 1

    for i_level in range(num_resolution):
        for j_block in range(num_resblocks):
            # residual block
            value = f'down_{i_level}.block_{j_block}'
            key_map[m_idx] = value 
            m_idx += 1
            if all_resolutions[i_level] in attn_resolutions:
                # attn block
                value = f'down_{i_level}.attn_{j_block}'
                key_map[m_idx] = value
                m_idx += 1
        if i_level != num_resolution - 1:
            # downsample block
            value = f'down_{i_level}.downsample'
            key_map[m_idx] = value
            m_idx += 1
    
    value = 'mid.block_1'
    key_map[m_idx] = value
    
    m_idx += 1
    value = 'mid.attn_1'
    key_map[m_idx] = value 

    m_idx += 1
    value = 'mid.block_2'
    key_map[m_idx] = value
    
    m_idx += 1
    for i_level in reversed(range(num_resolution)):
        for j_block in range(num_resblocks + 1):
            # residual block
            value = f'up_{i_level}.block_{j_block}'
            key_map[m_idx] = value
            m_idx += 1

            if all_resolutions[i_level] in attn_resolutions:
                # attn block
                value = f'up_{i_level}.attn_{j_block}'
                key_map[m_idx] = value
                m_idx += 1
        if i_level != 0:
            # upsample block
            value = f'up_{i_level}.upsample'
            key_map[m_idx] = value
            m_idx += 1
    value = 'norm_out'
    key_map[m_idx] = value

    m_idx += 1
    value = 'conv_out'
    key_map[m_idx] = value

    m_idx +=1
    return key_map


def load_dense(state_dict, sub_key, param):
    if sub_key == 'weight':
        weights = torch.from_numpy(state_dict['kernel']).T
    elif sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    param.copy_(weights)


def load_group_norm(state_dict, sub_key, param):
    if sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    elif sub_key == 'weight':
        weights = torch.from_numpy(state_dict['scale'])
    else:
        raise NotImplementedError(sub_key)
    param.copy_(weights.reshape(param.shape))


def load_conv(state_dict, sub_key, param):
    if sub_key == 'weight':
        weights = torch.from_numpy(state_dict['kernel'])
        # kernel: (kernel_size, kernel_size, C_in, C_out)
        # target: (C_out, C_in, kernel_size, kernel_size)
        weights = weights.permute(3, 2, 0, 1)
    elif sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    num_ch = param.shape[0]
    param.copy_(weights[:num_ch])


def load_nin(state_dict, sub_key, param):
    if sub_key == 'W':
        weights = torch.from_numpy(state_dict['kernel']).squeeze()
    elif sub_key == 'b':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    param.copy_(weights.reshape(param.shape))



def load_res_block(state_dict,
                  block_key, sub_key, 
                  param):
    res_dict = {
        'GroupNorm_0': 'norm1',
        'Conv_0': 'conv1',
        'Dense_0': 'temb_proj', 
        'GroupNorm_1': 'norm2', 
        'Conv_1': 'conv2', 
        'NIN_0': 'nin_shortcut'
    }
    module_key = res_dict[block_key]
    module_dict = state_dict[module_key]

    if module_key in ['conv1', 'conv2']:
        load_conv(state_dict=module_dict, 
                  sub_key=sub_key, 
                  param=param)
    elif module_key in ['norm1', 'norm2']:
        load_group_norm(state_dict=module_dict, 
                        sub_key=sub_key, 
                        param=param)
    elif module_key == 'temb_proj':
        load_dense(state_dict=module_dict, 
                   sub_key=sub_key, 
                   param=param)
    else:
        load_nin(state_dict=module_dict, 
                 sub_key=sub_key, 
                 param=param)

    
def load_attn_block(state_dict, 
                    block_key, sub_key, 
                    param):
    attn_dict = {
        'NINs': 'qkv', 
        'NIN_3': 'proj_out', 
        'GroupNorm_0': 'norm'
    }

    qkv_sub_map = {
        'W': 'kernel', 
        'b': 'bias'
    }
    module_key = attn_dict[block_key]

    if module_key == 'norm':
        load_group_norm(state_dict=state_dict[module_key],
                        sub_key=sub_key, 
                        param=param)
    elif module_key == 'proj_out':
        load_nin(state_dict=state_dict[module_key], 
                 sub_key=sub_key, 
                 param=param)
    else:
        # load qkv
        qkv_sub_key = qkv_sub_map[sub_key]

        weight_list = []
        sub_shape = list(param.shape)
        sub_shape[-1] = sub_shape[-1] // 3
        for qkv_key in module_key:
            weight = torch.from_numpy(state_dict[qkv_key][qkv_sub_key]).reshape(sub_shape)
            weight_list.append(weight)
        qkv_weight = torch.cat(weight_list, dim=-1)
        param.copy_(qkv_weight)



def load_pkd_ckpt(model, state_dict, config):
    key_map = get_key_map(config)
    print(key_map)
    # print('start')
    with torch.no_grad():
        for name, param in model.named_parameters():

            m = re.search(r'all_modules\.(\d+)\.(\S+)', name)
            module_id = int(m.group(1))
            sub_key = m.group(2)

            main_key = key_map[module_id]
        
            module_dict = state_dict[main_key]
            if 'attn' in main_key:
                # Attention block
                m = re.search(r'all_modules\.(\d+)\.(\S+)\.(\S+)', name)
                block_key = m.group(2)
                sub_key = m.group(3)
                load_attn_block(state_dict=module_dict, 
                                block_key=block_key, 
                                sub_key=sub_key, 
                                param=param)
            elif main_key == 'norm_out':
                # GroupNorm
                load_group_norm(state_dict=module_dict, 
                                sub_key=sub_key, 
                                param=param)
            elif main_key in ['conv_in', 'conv_out']:
                load_conv(state_dict=module_dict, 
                          sub_key=sub_key, 
                          param=param)
            elif main_key in ['dense0', 'dense1', 'class_emb']:
                load_dense(state_dict=module_dict, 
                           sub_key=sub_key, 
                           param=param)
            else:
                m = re.search(r'all_modules\.(\d+)\.(\S+)\.(\S+)', name)
                block_key = m.group(2)
                sub_key = m.group(3)
                load_res_block(state_dict=module_dict,
                                block_key=block_key, 
                                sub_key=sub_key, 
                                param=param)
            # print(f'Transfer from {main_key} to {name}')
    print('complete weights transfer')


def load_from_jax_ckpt(model, state_dict, config):
    key_map = build_state_map(config)
    # print(key_map)
    # print('start')
    with torch.no_grad():
        for name, param in model.named_parameters():

            m = re.search(r'all_modules\.(\d+)\.(\S+)', name)
            module_id = int(m.group(1))
            sub_key = m.group(2)

            main_key = key_map[module_id]
            if 'time_conv' in main_key:
                # print(f'Skipping time conv block: {main_key}')
                continue
            
            module_dict = state_dict[main_key]
            if 'attn' in main_key:
                # Attention block
                m = re.search(r'all_modules\.(\d+)\.(\S+)\.(\S+)', name)
                block_key = m.group(2)
                sub_key = m.group(3)
                load_attn_block(state_dict=module_dict, 
                                block_key=block_key, 
                                sub_key=sub_key, 
                                param=param)
            elif main_key == 'norm_out':
                # GroupNorm
                load_group_norm(state_dict=module_dict, 
                                sub_key=sub_key, 
                                param=param)
            elif main_key in ['conv_in', 'conv_out']:
                load_conv(state_dict=module_dict, 
                          sub_key=sub_key, 
                          param=param)
            elif main_key in ['dense0', 'dense1', 'class_emb']:
                load_dense(state_dict=module_dict, 
                           sub_key=sub_key, 
                           param=param)

            else:
                m = re.search(r'all_modules\.(\d+)\.(\S+)\.(\S+)', name)
                block_key = m.group(2)
                sub_key = m.group(3)
                load_res_block(state_dict=module_dict,
                                block_key=block_key, 
                                sub_key=sub_key, 
                                param=param)
            # print(f'Transfer from {main_key} to {name}')
    print('complete weights transfer')