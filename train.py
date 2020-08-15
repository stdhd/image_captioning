import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from joeynmt.decoders import RecurrentDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.metrics import bleu
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm, trange

import metrics
from metrics import bleu

from data import Flickr8k
from model import Image2Caption, Encoder
from visualize import Tensorboard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    embed_size = 128
    hidden_size = 512
    batch_size = 8
    fix_length = 18

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, fix_length=fix_length)
    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=os.cpu_count())  # set num_workers=0 for debugging

    data_dev = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.devImages.txt', 'data/Flickr8k.token.txt', transform=transform, fix_length=fix_length)
    dataloader_dev = DataLoader(data_dev, batch_size, num_workers=os.cpu_count())

    encoder = Encoder(models.mobilenet_v2, pretrained=True)
    vocab_size = len(data_train.corpus.vocab.itos)

    embeddings = Embeddings(embedding_dim=embed_size, vocab_size=vocab_size)
    decoder = RecurrentDecoder(rnn_type="lstm",
                               emb_size=embed_size,
                               hidden_size=hidden_size,
                               encoder=encoder,
                               vocab_size=vocab_size,
                               init_hidden='bridge',
                               attention='bahdanau' # or: 'luong'
                               )

    model = Image2Caption(encoder, decoder, embeddings, device).to(device)

    tensorboard = Tensorboard(device=device)
    tensorboard.add_images_with_ground_truth(data_dev)

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

            if i != 0 and i % 5 == 0:
                training_loss = running_loss / 5
                running_loss = 0.0

                tensorboard.writer.add_scalars('loss', {"train_loss": training_loss}, epoch * len(dataloader_train) + i)
                tensorboard.writer.flush()

        with torch.no_grad():
            loss_sum = 0
            sum_bleu_1 = 0
            sum_bleu_2 = 0
            sum_bleu_3 = 0
            sum_bleu_4 = 0
            for data in tqdm(dataloader_dev):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs, hidden, _, _ = model(inputs, labels)
                token_ids = torch.argmax(outputs.squeeze(0), dim=-1).cpu().detach().numpy()
                label_ids = labels.cpu().detach().numpy()

                for j in range(outputs.size(0)):
                    bleu_hypo = ' '.join([data_dev.corpus.vocab.itos[id] for id in token_ids[j].tolist()])
                    bleu_ref = [' '.join([data_dev.corpus.vocab.itos[id] for id in label_ids[j].tolist()])]
                    sum_bleu_1 += metrics.bleu(bleu_hypo, bleu_ref, 1)
                    sum_bleu_2 += metrics.bleu(bleu_hypo, bleu_ref, 2)
                    sum_bleu_3 += metrics.bleu(bleu_hypo, bleu_ref, 3)
                    sum_bleu_4 += metrics.bleu(bleu_hypo, bleu_ref, 4)

                log_probs = F.log_softmax(outputs, dim=-1)
                targets = labels.contiguous().view(-1)
                loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
                loss_sum += loss.item()

            # Add bleu score to board
            tensorboard.writer.add_scalars('loss', {"dev_loss": loss_sum / len(dataloader_dev)},
                                           (epoch + 1) * len(dataloader_train))
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-1": sum_bleu_1 / len(dataloader_dev)},
                                           (epoch + 1) * len(dataloader_train))
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-2": sum_bleu_2 / len(dataloader_dev)},
                                           (epoch + 1) * len(dataloader_train))
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-3": sum_bleu_3 / len(dataloader_dev)},
                                           (epoch + 1) * len(dataloader_train))
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-4": sum_bleu_4 / len(dataloader_dev)},
                                           (epoch + 1) * len(dataloader_train))
            # Add predicted text to board
            tensorboard.add_predicted_text((epoch + 1) * len(dataloader_train), data_dev, model)
            tensorboard.writer.flush()


            # TODO: Save model, if validation got better
            tensorboard.writer.flush()
