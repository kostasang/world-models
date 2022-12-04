import numpy as np, torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Class that implements dataset for training the V model
    """

    def __init__(self, storage_folder):
        """Initiliazes ImageDataset object"""
        sequences = np.load(storage_folder+'/trajectories.npy', allow_pickle=True)
        # Concatenate all sequencies in one dataset
        self.images = np.concatenate(sequences)
        self.images = torch.Tensor(self.images)
        # Place channels to correct potition
        self.images = self.images.permute(dims=(0,3,1,2))
        # Normalize pixel values
        self.images  = self.images / 255
        # Shuffle the first dimension
        idxs = np.arange(0, self.images.shape[0])
        np.random.shuffle(idxs)
        self.images = self.images[idxs]
    
    def __len__(self):
        """Returns number of instances in dataset"""
        return self.images.shape[0]

    def __getitem__(self, idx):
        """Returns dataset entry specified by idx"""
        return self.images[idx]

    
