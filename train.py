from datetime import datetime
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
from nltk.translate.bleu_score import corpus_bleu

from data import Flickr8k
from model import Image2Caption, Encoder
from visualize import Tensorboard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PRINT_TRAINING_LOSS_EVERY = 200

if __name__ == '__main__':
    embed_size = 128
    hidden_size = 512
    batch_size = 8
    fix_length = 18
    model_name = f'vgg16_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, fix_length=fix_length, max_vocab_size=10_000)
    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=os.cpu_count())  # set num_workers=0 for debugging

    data_dev = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.devImages.txt', 'data/Flickr8k.token.txt', transform=transform, fix_length=fix_length, max_vocab_size=10_000)
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

    tensorboard = Tensorboard(log_dir=f'runs/{model_name}', device=device)
    tensorboard.add_images_with_ground_truth(data_dev)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    last_validation_score = float('-inf')

    for epoch in trange(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_train)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
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

            if i != 0 and i % PRINT_TRAINING_LOSS_EVERY == 0:
                training_loss = running_loss / PRINT_TRAINING_LOSS_EVERY
                running_loss = 0.0

                tensorboard.writer.add_scalars('loss', {"train_loss": training_loss}, epoch * len(dataloader_train) + i)
                tensorboard.writer.flush()

        with torch.no_grad():
            loss_sum = 0
            bleu_1 = 0
            bleu_2 = 0
            bleu_3 = 0
            bleu_4 = 0
            for data in tqdm(dataloader_dev):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, image_names = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs, hidden, _, _ = model(inputs, labels)
                token_ids = torch.argmax(outputs.squeeze(0), dim=-1).cpu().detach().numpy()
                label_ids = labels.cpu().detach().numpy()

                bleu_references = []
                for image_name in image_names:
                    bleu_references.append(data_dev.get_all_references_for_image_name(image_name))  # TODO ignore id 1,2 and 3 ???

                bleu_hypotheses = []
                for j in range(label_ids.shape[0]):
                    refs = token_ids[j].tolist()
                    img_captions = [w for w in refs if w not in {1, 2, 3}]
                    bleu_hypotheses.append(img_captions)

                bleu_1 += corpus_bleu(bleu_references, bleu_hypotheses, weights=(1, 0, 0, 0))
                bleu_2 += corpus_bleu(bleu_references, bleu_hypotheses, weights=(0, 1, 0, 0))
                bleu_3 += corpus_bleu(bleu_references, bleu_hypotheses, weights=(0, 0, 1, 0))
                bleu_4 += corpus_bleu(bleu_references, bleu_hypotheses, weights=(0, 0, 0, 1))

                log_probs = F.log_softmax(outputs, dim=-1)
                targets = labels.contiguous().view(-1)
                loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
                loss_sum += loss.item()

            global_step = (epoch + 1) * len(dataloader_train) - 1
            # Add bleu score to board
            tensorboard.writer.add_scalars('loss', {"dev_loss": loss_sum / len(dataloader_dev)}, global_step)
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-1": bleu_1 / len(dataloader_dev)}, global_step)
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-2": bleu_2 / len(dataloader_dev)}, global_step)
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-3": bleu_3 / len(dataloader_dev)}, global_step)
            tensorboard.writer.add_scalars('bleu_validation', {"bleu-4": bleu_4 / len(dataloader_dev)}, global_step)
            # Add predicted text to board
            tensorboard.add_predicted_text(global_step, data_dev, model)
            tensorboard.writer.flush()

            # Save model, if score got better
            compared_score = bleu_1 / len(dataloader_dev)
            if last_validation_score < compared_score:
                last_validation_score = compared_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_sum / len(dataloader_dev),
                }, f'saved_models/{model_name}-bleu_1-{last_validation_score}.pth')
