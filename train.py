import os

import torch
import torch.nn.functional as F
from joeynmt.constants import PAD_TOKEN
from joeynmt.embeddings import Embeddings
from torch import optim, nn
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm, trange

from custom_decoder import CustomRecurrentDecoder
from data import Flickr8k
from model import Image2Caption, Encoder
from visualize import Tensorboard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.cuda.get_device_name())

    embed_size = 512
    hidden_size = 512
    batch_size = 16
    model_name = f'paper'

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=10_000)
    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=os.cpu_count())  # set num_workers=0 for debugging

    data_dev = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.devImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=10_000)
    dataloader_dev = DataLoader(data_dev, batch_size, num_workers=os.cpu_count())  # os.cpu_count()

    encoder = Encoder(models.vgg16, pretrained=True)
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
        hidden_dropout=0.5,
        emb_dropout=0.5
    )

    model = Image2Caption(encoder, decoder, embeddings, device).to(device)

    tensorboard = Tensorboard(log_dir=f'runs/{model_name}', device=device)
    tensorboard.add_images_with_ground_truth(data_dev)

    #criterion = nn.NLLLoss(ignore_index=data_train.corpus.vocab.stoi[PAD_TOKEN])
    criterion = nn.CrossEntropyLoss(ignore_index=data_train.corpus.vocab.stoi[PAD_TOKEN])
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
            outputs, _, att_probs, _ = model(inputs, labels,
                                             scheduled_sampling=False,
                                             batch_no=epoch * len(dataloader_train) + i,
                                             k=100,
                                             embeddings=embeddings)

            targets = labels[:, 1:].contiguous().view(-1)
            loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), targets.long())
            loss += 1. * ((1. - att_probs.sum(dim=1)) ** 2).mean()  # Doubly stochastic attention regularization
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        tensorboard.writer.add_scalars('loss', {"train_loss": running_loss / len(dataloader_train)}, epoch)
        tensorboard.writer.flush()

        with torch.no_grad():
            model.eval()

            loss_sum = 0
            bleu_1 = [0]
            bleu_2 = [0]
            bleu_3 = [0]
            bleu_4 = [0]
            for data in tqdm(dataloader_dev):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, image_names = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs, _, att_probs, _ = model(inputs, labels)
                targets = labels[:, 1:].contiguous().view(-1)  # shifted by one because of BOS
                loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), targets.long())
                loss += 1. * ((1. - att_probs.sum(dim=1)) ** 2).mean()  # Doubly stochastic attention regularization
                loss_sum += loss.item()

                for beam_size in range(1, len(bleu_1) + 1):
                    prediction, _ = model.predict(data_dev, inputs, data_dev.max_length, beam_size)
                    decoded_prediction = data_dev.corpus.vocab.arrays_to_sentences(prediction)

                    decoded_references = []
                    for image_name in image_names:
                        decoded_references.append(data_dev.corpus.vocab.arrays_to_sentences(data_dev.get_all_references_for_image_name(image_name)))

                    idx = beam_size - 1
                    bleu_1[idx] += bleu_score(decoded_prediction, decoded_references, max_n=1, weights=[1])
                    bleu_2[idx] += bleu_score(decoded_prediction, decoded_references, max_n=2, weights=[0.5] * 2)
                    bleu_3[idx] += bleu_score(decoded_prediction, decoded_references, max_n=3, weights=[1 / 3] * 3)
                    bleu_4[idx] += bleu_score(decoded_prediction, decoded_references, max_n=4, weights=[0.25] * 4)

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
