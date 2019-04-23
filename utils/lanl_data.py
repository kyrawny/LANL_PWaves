from __future__ import print_function, with_statement, division
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import torch.utils.data as data

class LANL_FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None, train_file='./data/train.csv', test_dir='./data/test/', sample_file='./data/sample_submission.csv'):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = train_file
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv(sample_file)
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, test_dir + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.int16})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y[-1]
        feature_dict['segment']
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
    
class LANL_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df):
        'Initialization'
        self.seg_id = df['seg_id']
        self.X = df['segment']
        self.y = df['target']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        seg_id = self.seg_id[index]
        X = torch.from_numpy(self.X[index]).float().unsqueeze(0)
        y = torch.from_numpy(np.array([self.y[index]])).float()
        sample = {'seg_id': seg_id, 'X': X, 'y': y}

        return sample
    
class LANL_Dataset_LR(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df):
        'Initialization'
        self.seg_id = df['seg_id']
        self.X = df['segment']
        self.y = df['target']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        seg_id = self.seg_id[index]
        X = torch.from_numpy(self.X[index]).float().unsqueeze(0)
        y = torch.from_numpy(np.array([self.y[index]])).float()
        sample = (X, y)

        return sample