import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joeynmt.decoders import RecurrentDecoder
from joeynmt.embeddings import Embeddings
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm, trange

from data import Flickr8k
from model import Image2Caption, Encoder


def print_sequence(dataset: Flickr8k, seq: np.array):
    ids = np.argmax(seq, axis=-1)
    print(" ".join([dataset.corpus.vocab.itos[id] for id in ids]))


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def validation(model, dataloader_val):
    print(len(dataloader_val))
    with torch.no_grad():
        print("Begin validation")
        running_loss = 0
        for i, data in enumerate(tqdm(dataloader_val)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs, hidden, att_probs, att_vectors = model(inputs, labels)
            log_probs = F.log_softmax(outputs, dim=-1)
            targets = labels.contiguous().view(-1)
            loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
            running_loss += loss.item()
        print(running_loss / len(dataloader_val))
        # TODO: BLEU Score


if __name__ == '__main__':
    embed_size = 128
    hidden_size = 512
    batch_size = 8
    fix_length = 18

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt',

                          transform=transform, fix_length=fix_length)
    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=os.cpu_count())  # set num_workers=0 for debugging

    encoder = Encoder(models.mobilenet_v2, pretrained=True)
    vocab_size = len(data_train.corpus.vocab.itos)

    embeddings = Embeddings(embedding_dim=embed_size, vocab_size=vocab_size)
    decoder = RecurrentDecoder(rnn_type="lstm",
                               emb_size=embed_size,
                               hidden_size=hidden_size,
                               encoder=encoder,
                               vocab_size=vocab_size,
                               init_hidden='bridge')

    model = Image2Caption(encoder, decoder, embeddings, device).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in trange(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_train)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden, att_probs, att_vectors = model(inputs, labels)
            log_probs = F.log_softmax(outputs, dim=-1)
            targets = labels.contiguous().view(-1)
            loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i != 0 and i % 50 == 0:
                print_sequence(data_train, outputs.cpu().detach().numpy()[0])
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
