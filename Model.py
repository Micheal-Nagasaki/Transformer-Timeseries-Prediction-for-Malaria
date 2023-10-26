# -*- coding: utf-8 -*-
"""
Title: Transformer Model for Time Series Prediction

@author: Micheal
"""

import torch
import torch.nn as nn
from icecream import ic  # For debugging purposes


class Transformer(nn.Module):
    def __init__(self, feature_size=7, num_layers=3, dropout=0):
        super(Transformer, self).__init__()

        # Define the encoder layer
        # This consists of a multi-head self-attention mechanism followed by a simple, position-wise fully connected feed-forward network.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,  # The number of expected features in the input
            nhead=7,  # The number of heads in the multihead attention models
            dropout=dropout  # The dropout value (default=0.5)
        )

        # Define the transformer encoder
        # This is a stack of N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers  # The number of sub-encoder-layers in the encoder
        )

        # Define the decoder
        # A linear layer to map the output of the transformer encoder to a single value
        self.decoder = nn.Linear(
            feature_size,  # The number of expected features in the input
            1  # The number of features in the output
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize the weights and biases of the decoder to ensure that they are neither too small nor too large
        initrange = 0.1
        self.decoder.bias.data.zero_()  # Set biases to zero
        self.decoder.weight.data.uniform_(-initrange, initrange)  # Set weights to random values in the range [-0.1, 0.1]

    def _generate_square_subsequent_mask(self, sz):
        # Generate a mask to ensure that the self-attention mechanism only attends to positions that precede or are at the current position in the sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        # The forward propagation method
        # src: the input tensor
        # device: the device on which to perform computations

        # Generate a mask and move it to the specified device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        
        # Pass the input tensor through the transformer encoder
        # This will apply a series of self-attention and feed-forward operations to the input tensor
        output = self.transformer_encoder(src, mask)
        
        # Pass the output of the transformer encoder through the decoder to produce the final output
        output = self.decoder(output)
        return output

# Usage:
# Instantiate the transformer model and move it to the appropriate device (CPU or GPU)
# transformer_model = Transformer().to(device)
# Forward propagate an input tensor through the model
# output = transformer_model(input_tensor, device)
