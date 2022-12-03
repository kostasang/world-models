import torch, numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SequenceDataset(Dataset):
    """
    Class that implements the dataset for training the M-model
    """

    def __init__(
        self, 
        storage_folder: str,
        visual_encoder: torch.nn.Module
    ):
        """Initializes SequenceDataset object"""
        # Load sequences of observations
        self.obs_sequences = np.load(storage_folder+'/trajectories.npy', allow_pickle=True)
        self.obs_sequences = [torch.Tensor(obs_sequence).permute(0,3,1,2)/255 for obs_sequence in tqdm(self.obs_sequences)]
        # Load sequences of actions
        self.action_sequences = np.load(storage_folder+'/actions.npy', allow_pickle=True)
        self.action_sequences = [torch.Tensor(action_seq).unsqueeze(-1) for action_seq in self.action_sequences]
        # Encode sequences of images using the already trained V-model
        self.encoded_sequences = self.__encode_observations(visual_encoder)
        # Get lengths of sequencies before performing padding
        self.lengths = [len(sequence) for sequence in self.encoded_sequences]
        # Pad the sequences so that the have the same length
        self.encoded_sequences = pad_sequence(self.encoded_sequences, batch_first=True).detach()
        self.action_sequences = pad_sequence(self.action_sequences, batch_first=True, padding_value=-1).detach()
        # Now keep from second to last for labels and from first to before-last for features
        self.targets = self.encoded_sequences[:,1:,:]
        self.encoded_sequences = self.encoded_sequences[:,0:-1,:]
        self.action_sequences = [seq[0:-1] for seq in self.action_sequences]
        self.lengths = [length -1 for length in self.lengths]

    @torch.inference_mode()
    def __encode_observations(self, visual_encoder: torch.nn.Module):
        """Encodes observed images to encoder model's latent space"""
        return [visual_encoder(sequence) for sequence in self.obs_sequences]
    
    def __len__(self):
        """Returns dataset's length"""
        return len(self.obs_sequences)

    def __getitem__(self, idx):
        """Get dataset entry specified by idx"""
        return self.encoded_sequences[idx], self.action_sequences[idx], self.targets[idx], self.lengths[idx]