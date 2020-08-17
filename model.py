from typing import Callable

import torch
import torch.nn as nn
from joeynmt.search import greedy
from torch import Tensor


class Image2Caption(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, embeddings: nn.Module, device: str, freeze_encoder: bool = True):
        super(Image2Caption, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embeddings = embeddings
        self.device = device

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        x = self.encoder(x)
        outputs, hidden, att_probs, att_vectors = self.decoder(
            trg_embed=self.embeddings(y.long()),
            encoder_output=x,
            encoder_hidden=x.mean(dim=1),
            src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
            unroll_steps=y.shape[1]
        )

        return outputs, hidden, att_probs, att_vectors

    def predict(self, x: Tensor, beam_size: int = 1):
        x = self.encoder(x)
        output, attention_scores = greedy(
            encoder_hidden=x.mean(dim=1),
            encoder_output=x,
            eos_index=3,
            src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
            embed=self.embeddings,
            bos_index=2,
            decoder=self.decoder,
            max_output_length=18)

        return output


class Encoder(nn.Module):
    def __init__(self, base_arch: Callable, pretrained=True):
        super(Encoder, self).__init__()
        loaded_model = base_arch(pretrained)
        self.features = loaded_model.features[:-1]  # drop MaxPool2d-layer
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))  # allow input images of variable size (14×14×512 as in paper 4.3)

        self.output_size = self.avgpool(self.features(torch.zeros(1, 3, 224, 224))).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)  # 512×14×14
        x = x.view(x.shape[0], x.shape[1], -1)  # 512×196
        x = x.permute(0, 2, 1)  # 196×512
        return x
