import os, glob
import pickle
import random
from collections import defaultdict

import math
import numpy as np

from torch.utils.data import get_worker_info


class Dataset10:
    def __init__(self, dsroot, split, batch_size):
        # check the split
        valid_splits = ["train", "val"]
        if split not in valid_splits:
            raise ValueError(f"invalid dataset split: {split} is not in {valid_splits}")
        
        # load the metadata
        meta_file = os.path.join(dsroot, "batches.meta")
        with open(meta_file, 'rb') as fh:
            meta = pickle.load(fh, encoding='bytes')
        self._class_names = [l.decode('utf-8') for l in meta[b'label_names']]
        
        # load the data
        if split == "val":
            file_glob = os.path.join(dsroot, "test_batch")
        else:
            file_glob = os.path.join(dsroot, "data_batch_?")

        entries = glob.glob(file_glob)
        entries.sort()  # for repeatability

        data = []
        for idx, data_file in enumerate(entries):
            with open(data_file, 'rb') as fh:
                raw_data = pickle.load(fh, encoding='bytes')
            
            names = raw_data[b'filenames']
            labels = raw_data[b'labels']
            images = raw_data[b'data']
            
            for name, label, image in zip(names, labels, images):
                name = name.decode('utf-8')
                image = np.moveaxis(np.reshape(image, (3, 32, 32)), 0, -1)
                data.append((name, label, image))
        
        self._data = data
        self._batch_size = batch_size
        
    def shuffle(self):
        random.shuffle(self._data)
        
    def num_classes(self):
        return len(self._class_names)
        
    def class_name(self, l):
        return self._class_names[l]

    def data_norm(self):
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.20220212, 0.19931542, 0.20086346)
        return (mean, std)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        # determine range of images for this worker
        worker_id = 0
        num_workers = 1
        if worker_info := get_worker_info():
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        num_batches = math.ceil(len(self)/self._batch_size)
        batches_per_worker = [len([i for i in range(x, num_batches, num_workers)]) for x in range(num_workers)]
        
        start_batch = sum([batches_per_worker[i] for i in range(worker_id)])
        end_batch = start_batch + batches_per_worker[worker_id]
        
        start_idx = start_batch * self._batch_size
        end_idx = min(end_batch * self._batch_size, len(self))

        # iterate over range
        for idx in range(start_idx, end_idx):
            name, label, image = self._data[idx]
            item = {
                'name': name,
                'label': label,
                'image': image
            }
            yield item


class Dataset100:
    def __init__(self, dsroot, split, batch_size):
        # check the split
        valid_splits = ["train", "val"]
        if split not in valid_splits:
            raise ValueError(f"invalid dataset split: {split} is not in {valid_splits}")
        
        # load the metadata
        meta_file = os.path.join(dsroot, "meta")
        with open(meta_file, 'rb') as fh:
            meta = pickle.load(fh, encoding='bytes')
        self._class_names = [l.decode('utf-8') for l in meta[b'fine_label_names']]
        
        # load the data
        data = []
        
        data_file = "train" if split == "train" else "test"
        data_file = os.path.join(dsroot, data_file)
        with open(data_file, 'rb') as fh:
            raw_data = pickle.load(fh, encoding='bytes')
        
        names = raw_data[b'filenames']
        labels = raw_data[b'fine_labels']
        images = raw_data[b'data']
        
        for name, label, image in zip(names, labels, images):
            name = name.decode('utf-8')
            image = np.moveaxis(np.reshape(image, (3, 32, 32)), 0, -1)
            data.append((name, label, image))
        
        self._data = data
        self._batch_size = batch_size
        
    def shuffle(self):
        random.shuffle(self._data)
        
    def num_classes(self):
        return len(self._class_names)
        
    def class_name(self, l):
        return self._class_names[l]
    
    def data_norm(self):
        mean = (0.50707516, 0.48654887, 0.44091784)
        std = (0.20079844, 0.19834627, 0.20219835)
        return (mean, std)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        # determine range of images for this worker
        worker_id = 0
        num_workers = 1
        if worker_info := get_worker_info():
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        num_batches = math.ceil(len(self)/self._batch_size)
        batches_per_worker = [len([i for i in range(x, num_batches, num_workers)]) for x in range(num_workers)]
        
        start_batch = sum([batches_per_worker[i] for i in range(worker_id)])
        end_batch = start_batch + batches_per_worker[worker_id]
        
        start_idx = start_batch * self._batch_size
        end_idx = min(end_batch * self._batch_size, len(self))

        # iterate over range
        for idx in range(start_idx, end_idx):
            name, label, image = self._data[idx]
            item = {
                'name': name,
                'label': label,
                'image': image
            }
            yield item


def dataset(dsroot, *, split, batch_size):
    if os.path.exists(os.path.join(dsroot, "batches.meta")):
        dset = Dataset10(dsroot, split, batch_size)
    else:
        dset = Dataset100(dsroot, split, batch_size)
    return dset

