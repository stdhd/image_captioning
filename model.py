from typing import Callable

import torch
import torch.nn as nn
from joeynmt.decoders import RecurrentDecoder
from joeynmt.embeddings import Embeddings
from torchsummary import summary
from torchvision import models

from torch import Tensor, optim

from data import list_of_unique_words, Flickr8k


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

        trg_embed = self.embeddings(y.long())
        # unroll_steps = trg_input.size(1)
        outputs, hidden, att_probs, att_vectors = self.decoder(
            trg_embed,
            encoder_output=x,
            encoder_hidden=torch.zeros((x.shape[0], self.encoder.output_size)).to(self.device),
            src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
            unroll_steps=y.shape[1]
        )
        return outputs, hidden, att_probs, att_vectors


class Encoder(nn.Module):
    def __init__(self, base_arch: Callable, output_size, pretrained=True):
        super(Encoder, self).__init__()
        loaded_model = base_arch(pretrained)
        self.features = loaded_model.features[:-1]  # drop MaxPool2d-layer
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))  # allow input images of variable size (14×14×512 as in paper 4.3)

        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)  # 512×14×14
        x = x.view(x.shape[0], 512, -1)  # 512×196
        x = x.permute(0, 2, 1)  # 196×512
        return x


if __name__ == '__main__':
    encoder = Encoder(models.vgg16, pretrained=True)
    # summary(encoder, input_size=(3, 224, 224), device='cpu')
    unique_words_list = list_of_unique_words('data/Flickr8k.token.txt')
    vocab_size = len(unique_words_list)
    embeddings = Embeddings(embedding_dim=512, vocab_size=vocab_size)
    decoder = RecurrentDecoder(rnn_type="lstm",
                               emb_size=512,
                               hidden_size=512,
                               encoder=encoder,
                               vocab_size=vocab_size,
                               init_hidden='last')

    model = Image2Caption(encoder, decoder, embeddings)
