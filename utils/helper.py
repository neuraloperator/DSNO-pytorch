import os
import re
import argparse


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