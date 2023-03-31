import os
import lmdb

import numpy as np
import torch
from torch.utils.data import Dataset


def split_list(arr, num_parts=4):
    data_dict = {}
    chunk_size = len(arr) // num_parts
    for i in range(num_parts):
        data_dict[i] = arr[i * chunk_size: (i + 1) * chunk_size]
    rem = len(arr) % num_parts
    for j in range(rem):
        data_dict[j].append(arr[num_parts * chunk_size + j])
    return data_dict

'''
This class is for loading trajectories from lmdb database

Args:
    - path: path to the lmdb database
    - data_shape: shape of each entry in the database: e.g. (C, T, H, W)
    - dims: dimensions to transpose the data such that it can be loaded as (T, C, H, W)
    - t_idx: time step indices to extract from the data
    - num_data: number of data to load
'''

class LMDBData(Dataset):
    def __init__(self, path, data_shape, dims, t_idx=None, num_data=None):
        super().__init__()
        env = lmdb.open(path, max_readers=32, 
                        readonly=True, readahead=False, 
                        lock=False, meminit=False)

        self.t_idx = t_idx
        self.dims = dims
        if num_data:
            self.length = num_data
        else:
            with env.begin(write=False) as txn:
                self.length = int(str(txn.get('length'.encode()), 'utf-8'))
        print(self.length)
        env.close()
        self.shape = data_shape
        if t_idx:
            self.use_idx = True
        else:
            self.use_idx = False
        self.path = path
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_db()

        key = f'{index}'.encode()
        value = self.txn.get(key)
        val = np.frombuffer(value, dtype=np.float32)
        extracted_val = self.extract(val)
        data = torch.from_numpy(extracted_val)
        return data

    def extract(self, value):
        data = value.reshape(self.shape)
        data = data.transpose(self.dims)
        if self.use_idx:
            data = data[self.t_idx]
        return data

    def open_db(self):
        self.env = lmdb.open(self.path, max_readers=32, 
                        readonly=True, readahead=False, 
                        lock=False, meminit=False)
        self.txn = self.env.begin(write=False, buffers=True)

'''
This class is for loading data from lmdb databases and labels from npy files. 
Args:
    - path: path to the lmdb database
    - data_shape: shape of each entry in the database: e.g. (C, T, H, W)
    - dims: dimensions to transpose the data such that it can be loaded as (T, C, H, W)
    - t_idx: time step indices to extract from the data
    - num_data: number of data to load
'''


class ImageNet(Dataset):
    def __init__(self, path, data_shape, dims, t_idx=None, num_data=None):
        super().__init__()

        self.db_dir = os.path.join(path, 'lmdb')
        label_path = os.path.join(path, 'labels.npy')

        self.t_idx = t_idx
        self.dims = dims
        self.labels = torch.from_numpy(np.load(label_path))

        if num_data:
            self.length = num_data
        else:
            self.length = self.labels.shape[0]
        self.shape = data_shape

        if t_idx:
            self.use_idx = True
        else:
            self.use_idx = False
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_db()

        key = f'{index}'.encode()
        value = self.txn.get(key)
        val = np.frombuffer(value, dtype=np.float32)
        extracted_val = self.extract(val)
        data = torch.from_numpy(extracted_val)
        return data, self.labels[index]

    def extract(self, value):
        data = value.reshape(self.shape)
        data = data.transpose(self.dims)
        if self.use_idx:
            data = data[self.t_idx]
        return data

    def open_db(self):
        self.env = lmdb.open(self.db_dir, max_readers=512, 
                        readonly=True, readahead=False, 
                        lock=False, meminit=False)
        self.txn = self.env.begin(write=False, buffers=True)