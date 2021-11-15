"""Dataloading for GFD data."""

import numpy as np
import torch
from torch.utils.data import dataset, dataloader, SequentialSampler

import utils
from functools import reduce
import pandas as pd

from typing import List

class CommoditiesDataSet(dataset.Dataset):
    """DataSet Loader for commodities GFD data"""

    __DATA_PATH__ = 'data/'
    def __init__(self, commodity: str = 'corn', seq_length: int  = 10, split = 'train') -> None:
        """Initialize the data loader."""
        super().__init__()
        self.seq_length = seq_length

        # read csvs and combine into single dataframe
        read_csv = pd.read_csv(f'data/{commodity}.csv', header = 2)
        self.data = utils.process_data(read_csv, 'close_price')
        self.data = self.data.sort_values(by='Date', ascending=True).reset_index(drop=True)

        # convert to log returns
        self.data['log_return'] = np.log(self.data['close_price']) - \
                           np.log(self.data['close_price'].shift(1))
        self.data = self.data[1:]

        # train/dev/test 
        if split == 'train':
            self.data = self.data.loc[(self.data['Date'] < '2018-01-01')]
        elif split == 'val':
            self.data = self.data.loc[(self.data['Date'] >= '2018-01-01')]
            self.data = self.data.loc[(self.data['Date'] < '2019-01-01')]

        elif split == 'test':
            self.data = self.data.loc[(self.data['Date'] >= '2019-01-01')]
            self.data = self.data.loc[(self.data['Date'] < '2020-01-01')]

        else:
            raise ValueError()
        self.data = self.data['log_return'].to_numpy()


    def __len__(self):
        """Return the total number of samples in the dataset."""  

        # adjust by sequence length     
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return a sample.
        A sequence consists of a training sample of length self.seq_length
        and corresponding label is the next day's return.
        """

        X = torch.tensor(self.data[index : index + self.seq_length], dtype = torch.double)
        y = torch.tensor(self.data[index + self.seq_length: index + self.seq_length + 1], dtype = torch.double)

        return X, y


def get_dataloader(commodity, seq_length, split, batch_size):
    dataset = CommoditiesDataSet(commodity, seq_length, split)
    return dataloader.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )