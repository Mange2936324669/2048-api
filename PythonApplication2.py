import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable

class My_dataset(Dataset):
    def __init__(self,csv_file,transform=None,target_transform=None):
        frame = pd.read_csv(csv_file)
        tmp = frame.values

        self.data = tmp[:,0:16]/12
        self.label = tmp[:,16]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idex):
        board = self.data[idex].reshape(4,4)
        board = board[:,:,np.newaxis]   #torch.from_numpy

        label = self.label[idex]
        label = label.astype(int)

        if self.transform is not None:
            board = self.transform(board)
        return board,label
