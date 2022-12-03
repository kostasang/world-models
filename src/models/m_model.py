import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence

class MModel(nn.Module):
    """
    Class implementing M-Model which is an LSTM
    network with a projection layer on top that predicts
    the next encoded state of the current encoded state and the
    performed action
    """

    def __init__(
        self,
        input_dim=128, 
        num_lstm_layers=1,
        hidden_size=256,
        dropout=0.3,
        output_dim=128,
        packing=True
    ):
        """Initializes LSTM model"""
        super(MModel, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )
        self.projection = nn.Linear(in_features=hidden_size, out_features=output_dim)
        self.packing = packing

    def forward(self, x, lengths):
        """Implement forward pass of the model"""
        if self.packing:
            initial_padded_length = x.shape[1]
            # Sort before packing
            sorted_idx = torch.argsort(lengths, descending=True)
            lengths = lengths[sorted_idx]
            x = x[sorted_idx]
            x = pack_padded_sequence(
                input=x,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=True
            )
        out, _ = self.lstm_layer(x)
        if self.packing:
            out, _ = pad_packed_sequence(
                sequence=out,
                batch_first=True,
                total_length=initial_padded_length
            )
            # Fix order back to normal
            restored_idx = torch.argsort(sorted_idx)
            out = out[restored_idx]
        return self.projection(out)