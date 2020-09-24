from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.decoders import Decoder
from joeynmt.embeddings import Embeddings
from joeynmt.search import greedy, beam_search
from torch import Tensor

from data import Flickr8k


class Image2Caption(nn.Module):
    """
    Class combining decoder and encoder
    """

    def __init__(self, encoder: nn.Module, decoder: Decoder, embeddings: Embeddings, device: str, freeze_encoder: bool = True, dropout_after_encoder=0):
        """
        Combined encoder-decoder model
        :param encoder: nn.Module object representing the encoder
        :param decoder: nn.Module object representing the decoder
        :param embeddings: joeynmt.embeddings.Embeddings object
        :param device: torch.device('cpu') or torch.device('cuda') for example
        :param freeze_encoder: If true, do not continue learning the encoder
        :param: dropout_after_encoder: If true, enable dropout layer between encoder and decoder
        """
        super(Image2Caption, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embeddings = embeddings
        self.device = device
        self.dropout_after_encoder_layer = nn.Dropout(dropout_after_encoder)

        # In case we do not want to continue training the encoder, gradient calculation is disabled for the encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor, **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Forward function to feed in images and true captions
        :param x: Image data as tensor (batch_size, 3, 224, 224)
        :param y: True labels as tensors of token numbers (batch_size, max_sequence_length)
        :param kwargs: Parameters to be passed on to the decoder's forward function
        :return:
            - outputs: Tensor of predicted tokens (batch_size, unroll_steps, vocab_size)
            - hidden: tensor of last hidden state (num_layers, batch_size, hidden_size)
            - att_probs: Attention probabilities of whole unrolling (batch_size, unroll_steps, src_length)
            - att_vectors: Attention vectors of whole unrolling (batch_size, unroll_steps, hidden_size)
        """
        kwargs['unroll_steps'] = kwargs.get('unroll_steps') - 1

        x = self.encoder(x)
        x = self.dropout_after_encoder_layer(x)
        outputs, hidden, att_probs, att_vectors = self.decoder(
            trg_embed=self.embeddings(y.long()),
            encoder_output=x,
            encoder_hidden=x.mean(dim=1),
            src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
            **kwargs
        )

        return outputs, hidden, att_probs, att_vectors

    def predict(self, data: Flickr8k, x: Tensor, max_output_length: int, beam_size: int = 1, beam_alpha: float = 0.4):
        """
        Predict cpation of given images, for inference only. This method allows beam search.
        :param data: Flickr8k object
        :param x: Image data as tensor (batch_size, 3, 224, 224)
        :param max_output_length: Length of the sequence after the generation is cut
        :param beam_size: Number of beams to use in search. For 1, greedy search is done.
        :param beam_alpha: Penalize length with alpha factor
        :return:
            - output: Tensor of predicted tokens (batch, unroll_steps, vocab_size)
            - attention_scores: Attention probabilities of whole unrolling (batch_size, unroll_steps, src_length)
        """
        x = self.encoder(x)

        if beam_size < 2:
            output, attention_scores = greedy(
                encoder_output=x, encoder_hidden=x.mean(dim=1),
                src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
                bos_index=data.corpus.vocab.stoi[BOS_TOKEN], eos_index=data.corpus.vocab.stoi[EOS_TOKEN],
                embed=self.embeddings,
                decoder=self.decoder,
                max_output_length=max_output_length
            )
        else:
            output, attention_scores = beam_search(
                size=beam_size,
                encoder_output=x, encoder_hidden=x.mean(dim=1),
                src_mask=torch.ones(x.shape[0], 1, x.shape[1]).byte().to(self.device),
                bos_index=data.corpus.vocab.stoi[BOS_TOKEN], eos_index=data.corpus.vocab.stoi[EOS_TOKEN], pad_index=data.corpus.vocab.stoi[PAD_TOKEN],
                embed=self.embeddings,
                decoder=self.decoder,
                alpha=beam_alpha,
                max_output_length=max_output_length
            )

        return output, attention_scores


class Encoder(nn.Module):
    def __init__(self, base_arch: Callable, pretrained=True):
        """
        Encoder using given classification model as feature extractor
        :param base_arch: Constructor of torchvision.models
        :param pretrained: Load pre-trained model state
        """
        super(Encoder, self).__init__()
        loaded_model = base_arch(pretrained)
        self.features = loaded_model.features[:-1]  # drop MaxPool2d-layer
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))  # allow input images of variable size (14×14×512 as in paper 4.3)

        self.output_size = self.avgpool(self.features(torch.zeros(1, 3, 224, 224))).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function of encoder
        :param x: Input images (batch_size, 3, 224, 224)
        :return: Extracted features (batch_size, features_n, output_size)
        """
        x = self.features(x)
        x = self.avgpool(x)  # 512×14×14
        x = x.view(x.shape[0], x.shape[1], -1)  # 512×196
        x = x.permute(0, 2, 1)  # 196×512
        return x
