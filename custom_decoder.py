from typing import Optional

import torch
from joeynmt.decoders import RecurrentDecoder
from torch import Tensor


class CustomRecurrentDecoder(RecurrentDecoder):
    def __init__(self, *args, **kwargs):
        super(CustomRecurrentDecoder, self).__init__(*args, **kwargs)

        encoder = kwargs.get('encoder')
        hidden_size = kwargs.get('hidden_size')

        self.bridge_layer_h = torch.nn.Linear(encoder.output_size, hidden_size, bias=True)
        self.bridge_layer_c = torch.nn.Linear(encoder.output_size, hidden_size, bias=True)

    def _init_hidden(self, encoder_final: Tensor = None) -> (Tensor, Optional[Tensor]):
        hidden_h = torch.tanh(self.bridge_layer(encoder_final)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        #hidden_c = torch.tanh(self.bridge_layer(encoder_final)).unsqueeze(0).repeat(self.num_layers, 1, 1)

        return hidden_h, hidden_h
