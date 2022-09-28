import torch
import torch.nn as nn

class VModel(nn.Module):
    """
    Class implementing V-model which is a VAE composed by
    a visual encoder model and a visual decoder model
    """

    def __init__(self, n_z=64):
        """Initializes VAE"""
        self.encoder = VisualEncoder(n_z=n_z)
        self.decoder = VisualDecoder(n_z=n_z)
        # TODO: sample latent vector using Gausian distribution

    def forward(self, x):
        "Implements forward pass of the model"
        z = self.encoder(x)
        return self.decoder(z)

    def extract_encoder(self):
        """Returns encoder part of VAE"""
        return self.encoder
    
    def extract_decoder(self):
        """Returns decoder part of VAE"""
        return self.decoder

class VisualEncoder(nn.Module):
    """
    Class for the encoder part of V model
    """
    
    def __init__(self, n_z=64):
        """Initiliazes V model encoder"""
        super(VisualEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.projection = nn.Linear(in_features=1024, out_features=n_z)

    def forward(self, x):
        """Implementes forward pass of the model"""
        out = self.conv(x)
        return self.projection(out)


class VisualDecoder(nn.Module):
    """
    Class for the decoder part of V model
    """

    def __init__(self, n_z=64):
        """Initilializes V model decoder"""
        super(VisualDecoder, self).__init__()
        self.deconv = nn.Sequential(            
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(in_features=n_z, out_features=1024)

    def forward(self, x):
        """Implementes forward pass of the model"""
        out = self.projection(x).unsqueeze(-1).unsqueeze(-1)
        return self.deconv(out)