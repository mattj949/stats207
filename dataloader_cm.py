"""Dataloading for GFD data."""

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import dataset, dataloader, SequentialSampler, RandomSampler

from functools import reduce
import pandas as pd

from typing import List

# Define a helper function to process the data
def process_data(df, label):
    
    # Convert the Date column from a string to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # We only want dates prior to this cutoff
    df = df.loc[(df['Date'] <= '2021-11-01')]
    
    # We only want dates after this cutoff
    df = df.loc[(df['Date'] >= '1980-01-01')]
    
    # Carries forward old prices, so we aren't using future information
    df.fillna(method = 'ffill', inplace=True) 
    
    # Drop the 'Ticker' column
    df = df.drop(columns = ['Ticker'], axis=1)
    
    # Rename the 'Close' column with the passed label
    df = df.rename(columns={'Close': label})
    #df = df.drop(columns = 'Close')
    
    return df

class CommoditiesExogenousDataSet(dataset.Dataset):
    """DataSet Loader for commodities GFD data"""

    __DATA_PATH__ = 'data/'
    def __init__(self, filepath=None, seq_length: int  = 20, split = 'train', pxcol = 'Close/Last', exog = ['Volume'], dateformat = '%m/%d/%Y',  ) -> None:
        """Initialize the data loader."""
        super().__init__()
        self.seq_length = seq_length

        # read csvs and combine into single dataframe
        read_csv = pd.read_csv(f'data/{filepath}.csv', header = 0)
        self.data = read_csv[['Date'] + [pxcol] + exog]
        self.data['Date'] = pd.to_datetime(self.data['Date'], format = dateformat)
        self.data = self.data.sort_values(by='Date', ascending=True)
        self.data.dropna(inplace = True)

        # convert to log returns
        self.data['log_return'] = np.log(self.data[pxcol]) - np.log(self.data[pxcol].shift(1))
        self.data = self.data[1:]
        self.alldata = self.data[['log_return'] + exog].to_numpy()
        # train/dev/test 
        if split == 'train':
            self.data = self.data.loc[(self.data['Date'] < '2018-01-01')]
        elif split == 'val':
            self.data = self.data.loc[(self.data['Date'] >= '2018-01-01')]
            self.data = self.data.loc[(self.data['Date'] < '2019-01-01')]
            assert(len(self.data) > 0)

        elif split == 'test':
            self.data = self.data.loc[(self.data['Date'] >= '2019-01-01')]
            self.data = self.data.loc[(self.data['Date'] < '2020-01-01')]

        else:
            raise ValueError()
        
        self.log_return = self.data[['log_return'] + exog].to_numpy()


    def __len__(self):
        """Return the total number of samples in the dataset."""  

        # adjust by sequence length     
        return len(self.log_return) - self.seq_length

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return a sample.
        A sequence consists of a training sample of length self.seq_length
        and corresponding label is the next day's return.
        """

        X = torch.tensor(self.log_return[index : index + self.seq_length], dtype = torch.double)
        mean = self.alldata[index + self.seq_length].mean(axis = 0)
        std = self.alldata[index + self.seq_length].std(axis = 0)
        if std == 0:
            # don't normalize
            X = X
        else:
        # normalization of X by training set mean and std dev
            X = (X - mean) / std
        y = torch.tensor(self.log_return[index + self.seq_length: index + self.seq_length + 1, 0], dtype = torch.double)
        y = torch.cat(((torch.sign(y) == 1.0).to(float), y))
        
        return X, y

class CommoditiesDataSet(dataset.Dataset):
    """DataSet Loader for commodities GFD data"""

    __DATA_PATH__ = 'data/'
    def __init__(self, commodity: str = 'corn', seq_length: int  = 10, split = 'train') -> None:
        """Initialize the data loader."""
        super().__init__()
        self.seq_length = seq_length

        # read csvs and combine into single dataframe
        read_csv = pd.read_csv(f'data/{commodity}.csv', header = 2)
        self.data = process_data(read_csv, 'close_price')
        self.data = self.data.sort_values(by='Date', ascending=True).reset_index(drop=True)

        # convert to log returns
        self.data['log_return'] = np.log(self.data['close_price']) - np.log(self.data['close_price'].shift(1))
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
        self.log_return = self.data['log_return'].to_numpy()
        


    def __len__(self):
        """Return the total number of samples in the dataset."""  

        # adjust by sequence length     
        return self.log_return.shape[0] - self.seq_length

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return a sample.
        A sequence consists of a training sample of length self.seq_length
        and corresponding label is the next day's return.
        """

        X = torch.tensor(self.log_return[index : index + self.seq_length], dtype = torch.double)

        # get the last observation and only the log return column
        y = torch.tensor(self.log_return[index + self.seq_length: index + self.seq_length + 1], dtype = torch.double)
        

        return X, y


def get_dataloader(commodity, seq_length, split, batch_size):
    dataset = CommoditiesDataSet(commodity, seq_length, split)
    return dataloader.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=RandomSampler(dataset),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
def get_dataloader_exog(filepath, seq_length, split, batch_size, pxcol = 'Close/Last', exog = ['Volume'], dateformat = '%m/%d/%Y'):
    dataset = CommoditiesExogenousDataSet(filepath, seq_length, split, pxcol = pxcol, exog = exog, dateformat=dateformat)
    if split == "train":
        sampler=RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return dataloader.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )