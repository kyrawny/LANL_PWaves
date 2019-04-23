from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class LANLDataset(Dataset):
    def __init__(self, mode, data, target=None):
        self.mode = mode
        self.data = data
        if mode == 'train':
            self.target = target

    def train_val_split(self, train_ratio, val_ratio):
        if (train_ratio + val_ratio != 1):
            raise Exception('Ratios should sum to one.')
        dataset_length = len(self.data)
        train_length = int(train_ratio * dataset_length)
        val_length = len(self) - train_length
        splits = [train_length, val_length]
        return random_split(self, splits)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        if isinstance(index, torch.Tensor):
            index = index.item()
        data = torch.Tensor(self.data[index])
        if self.mode == 'train':
            target = torch.Tensor([self.target[index]]).squeeze(0)
            sample = {'data': data, 'target': target}
        elif self.mode == 'test':
            sample = {'data':data}
        return sample

class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = './data/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('./data/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, './data/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.int16})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['segment'] = x
        feature_dict['seg_id'] = seg_id

        # create features here
        # for example:
        # feature_dict['mean'] = np.mean(x)

        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)
