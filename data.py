import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader



def data_loader(batch_size=1024, shuffle=True):
    """
    Creata data loaders from .npy files.

    Args:
        batch_size (int, optional): Batch size for training and testing. Defaults to 1024.
        shuffle (bool, optional): Shuffle data. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Dataloader for training and testing. 
    """
    # load numpy files and convert to torch.Tensor 
    X_train = torch.tensor(np.load('data/X_train.npy'))    
    X_test = torch.tensor(np.load('data/X_test.npy'))
    y_train = torch.tensor(np.load('data/y_train.npy'))
    y_test = torch.tensor(np.load('data/y_test.npy'))

    # create dataset and dataloader
    train_set = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    test_set = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader
