import torch
from torch.utils.data import DataLoader

from torchvision import transforms as T

import torch.nn as nn
import torch.nn.functional as F
from joeynmt.decoders import RecurrentDecoder
from joeynmt.embeddings import Embeddings
from torchvision import models

from torch import optim
from tqdm import tqdm, trange

from data import list_of_unique_words, Flickr8k
from model import Image2Caption, Encoder

import numpy as np


def print_sequence(dataset: Flickr8k, seq: np.array):
    wl = np.array(dataset.word_list)
    for batch in range (seq.shape[0]):
        ids = np.argmax(seq[batch], axis=-1)
        print(" ".join(wl[ids]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    embed_size = 128
    hidden_size = 512
    output_size = 512 # output size for encoder = input size for decoder
    max_sequence_size = 5
    batch_size = 1
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt',
                          transform=transform)
    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=0)

    encoder = Encoder(models.vgg16, output_size=output_size, pretrained=True)
    # summary(encoder, input_size=(3, 224, 224), device='cpu')
    unique_words_list = list_of_unique_words('data/Flickr8k.token.txt')
    vocab_size = len(unique_words_list)
    embeddings = Embeddings(embedding_dim=embed_size, vocab_size=vocab_size)
    decoder = RecurrentDecoder(rnn_type="lstm",
                               emb_size=embed_size,
                               hidden_size=hidden_size,
                               encoder=encoder,
                               vocab_size=vocab_size,
                               init_hidden='last')

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
            print_sequence(data_train, outputs.detach().numpy())
            log_probs = F.log_softmax(outputs, dim=-1)
            targets = labels.contiguous().view(-1)
            loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), targets.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

