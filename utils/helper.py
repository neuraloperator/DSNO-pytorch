import os
import re
import argparse
import requests


def count_params(model):
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_latest_ckpt(dir):
    latest_id = -1
    for file in os.listdir(dir):
        if file.endswith('.pt'):
            m = re.search(r'solver-model_(\d+)\.pt', file)
            if m:
                ckpt_id = int(m.group(1))
                latest_id = max(latest_id, ckpt_id)
        else:
            ckpt_id = -1
    if latest_id == -1:
        return -1
    else:
        ckpt_path = os.path.join(dir, f'solver-model_{latest_id}.pt')
        return ckpt_path
    

'''
Download pre-trained checkpoints from progressive distillation
'''

_url_dict = {
    'cifar_original': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/cifar_original', 
    'imagenet_original': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/imagenet_original',
    'imagenet_16': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/imagenet_16',
    'cifar_8': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/cifar_8',
}


def download_file(url, filepath):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)


def prepare_pretrain_file(filepath):
    if not os.path.exists(filepath):
        model_name = os.path.basename(filepath)
        url = _url_dict[model_name]
        print(f'Downloading {model_name} from {url} to {filepath}')
        download_file(url, filepath)