import os
from datetime import datetime

import torch
import torch.nn.functional as F
from joeynmt.constants import PAD_TOKEN
from joeynmt.embeddings import Embeddings
from nltk.translate.bleu_score import corpus_bleu
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm, trange

from custom_decoder import CustomRecurrentDecoder
from data import Flickr8k
from equal_sampler import EqualBatchSampler
from model import Image2Caption, Encoder
from visualize import Tensorboard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    embed_size = 512
    hidden_size = 512
    batch_size = 16
    model_name = f'mobilenet_v2_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=10_000)
    dataloader_train = DataLoader(data_train, num_workers=os.cpu_count(), batch_sampler=EqualBatchSampler(batch_size, True, data_train))  # set num_workers=0 for debugging

    data_dev = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.devImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=10_000)
    dataloader_dev = DataLoader(data_dev, batch_size, num_workers=os.cpu_count())  # os.cpu_count()

    encoder = Encoder(models.mobilenet_v2, pretrained=True)
    vocab_size = len(data_train.corpus.vocab.itos)

    embeddings = Embeddings(embedding_dim=embed_size, vocab_size=vocab_size)
    decoder = CustomRecurrentDecoder(
        rnn_type="lstm",
        emb_size=embed_size,
        hidden_size=hidden_size,
        encoder=encoder,
        vocab_size=vocab_size,
        init_hidden='bridge',
        attention='bahdanau',  # or: 'luong'
        hidden_dropout=0.2,
        emb_dropout=0.2
    )

    model = Image2Caption(encoder, decoder, embeddings, device).to(device)

    tensorboard = Tensorboard(log_dir=f'runs/{model_name}', device=device)
    tensorboard.add_images_with_ground_truth(data_dev)

    criterion = nn.NLLLoss(ignore_index=data_train.corpus.vocab.stoi[PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    last_validation_score = float('-inf')

    for epoch in trange(100):  # loop over the dataset multiple times
        model.train()

        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_train)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden, att_probs, att_vectors = model(inputs, labels,
                                                            scheduled_sampling=True,
                                                            batch_no=epoch*len(dataloader_train) + i,
                                                            k=100,
                                                            embeddings=embeddings)
            log_probs = F.log_softmax(outputs, dim=-1)
            targets = labels[:, 1:].contiguous().view(-1)
            loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        tensorboard.writer.add_scalars('loss', {"train_loss": running_loss / len(dataloader_train)}, epoch)
        tensorboard.writer.flush()

        with torch.no_grad():
            model.eval()

            loss_sum = 0
            bleu_1 = [0, 0, 0, 0, 0]
            bleu_2 = [0, 0, 0, 0, 0]
            bleu_3 = [0, 0, 0, 0, 0]
            bleu_4 = [0, 0, 0, 0, 0]
            for data in tqdm(dataloader_dev):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, image_names = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs, _, _, _ = model(inputs, labels)
                log_probs = F.log_softmax(outputs, dim=-1)
                targets = labels[:, 1:].contiguous().view(-1)  # shifted by one because of BOS
                loss = criterion(log_probs.contiguous().view(-1, log_probs.shape[-1]), targets.long())
                loss_sum += loss.item()

                for beam_size in range(1, len(bleu_1) + 1):
                    prediction, _ = model.predict(data_dev, inputs, data_dev.max_length, beam_size)
                    decoded_prediction = data_dev.corpus.vocab.arrays_to_sentences(prediction)

                    decoded_references = []
                    for image_name in image_names:
                        decoded_references.append(data_dev.corpus.vocab.arrays_to_sentences(data_dev.get_all_references_for_image_name(image_name)))

                    idx = beam_size - 1
                    bleu_1[idx] += corpus_bleu(decoded_references, decoded_prediction, weights=(1, 0, 0, 0))
                    bleu_2[idx] += corpus_bleu(decoded_references, decoded_prediction, weights=(0, 1, 0, 0))
                    bleu_3[idx] += corpus_bleu(decoded_references, decoded_prediction, weights=(0, 0, 1, 0))
                    bleu_4[idx] += corpus_bleu(decoded_references, decoded_prediction, weights=(0, 0, 0, 1))

            global_step = epoch
            # Add bleu score to board
            tensorboard.writer.add_scalars('loss', {"dev_loss": loss_sum / len(dataloader_dev)}, global_step)
            for idx in range(len(bleu_1)):
                tensorboard.writer.add_scalar(f'BEAM-{idx + 1}/BLEU-1', bleu_1[idx] / len(dataloader_dev), global_step)
                tensorboard.writer.add_scalar(f'BEAM-{idx + 1}/BLEU-2', bleu_2[idx] / len(dataloader_dev), global_step)
                tensorboard.writer.add_scalar(f'BEAM-{idx + 1}/BLEU-3', bleu_3[idx] / len(dataloader_dev), global_step)
                tensorboard.writer.add_scalar(f'BEAM-{idx + 1}/BLEU-4', bleu_4[idx] / len(dataloader_dev), global_step)
            # Add predicted text to board
            tensorboard.add_predicted_text(global_step, data_dev, model, data_dev.max_length)
            tensorboard.writer.flush()

            # Save model, if score got better
            compared_score = bleu_1[0] / len(dataloader_dev)
            if last_validation_score < compared_score:
                last_validation_score = compared_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_sum / len(dataloader_dev),
                }, f'saved_models/{model_name}-bleu_1-{last_validation_score}.pth')
