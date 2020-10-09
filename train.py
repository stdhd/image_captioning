import os
from typing import Tuple, Callable

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from joeynmt.constants import PAD_TOKEN
from joeynmt.decoders import TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.helpers import ConfigurationError
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm, trange

from custom_decoder import CustomRecurrentDecoder
from data import Flickr8k
from model import Image2Caption, Encoder
from pretrained_embeddings import PretrainedEmbeddings
from visualize import Tensorboard
from yaml_parser import parse_yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def clip_gradient(optimizer: Optimizer, grad_clip: float) -> None:
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def setup_model(params: dict, data: Flickr8k, pretrained_embeddings: PretrainedEmbeddings) -> Tuple[Embeddings, Image2Caption]:
    """
    setup embeddings and seq2seq model

    :param params: params from the yaml file
    :param data: Flickr Dataset class
    :param pretrained_embeddings: PretrainedEmbeddings if selected in yaml
    :return: word embeddings, seq2seq model
    """

    def get_base_arch(encoder_name: str) -> Callable:
        """
        wrapper for model, as EfficientNet does not support __name__

        :param encoder_name: name of the encoder to load
        :return: base_arch
        """
        if 'efficientnet' in encoder_name:
            base_arch = EfficientNet.from_pretrained(encoder_name).to(device)
            base_arch.__name__ = encoder_name
            return base_arch
        else:
            return getattr(models, encoder_name)

    encoder = Encoder(get_base_arch(params.get('encoder')), device, pretrained=True)
    vocab_size = len(data.corpus.vocab.itos) if not params.get('embed_pretrained', False) else 300

    if params.get('decoder_type', 'RecurrentDecoder') == 'RecurrentDecoder':
        decoder_type = CustomRecurrentDecoder
    else:
        decoder_type = TransformerDecoder

    decoder = decoder_type(
        rnn_type=params.get('rnn_type'),
        emb_size=params['embed_size'] if not params.get('embed_pretrained', False) else 300,
        hidden_size=params['hidden_size'],
        encoder=encoder,
        vocab_size=vocab_size,
        init_hidden='bridge',
        attention=params['attention'],
        hidden_dropout=params['hidden_dropout'],
        emb_dropout=params['emb_dropout'],
        num_layers=params.get('decoder-num_layers', 1)
    )

    if params.get('embed_pretrained', False):
        embeddings = pretrained_embeddings
    else:
        embeddings = Embeddings(embedding_dim=params['embed_size'], vocab_size=vocab_size)

    return embeddings, Image2Caption(encoder, decoder, embeddings, device, params['freeze_encoder'], params.get('dropout_after_encoder', 0), params['hidden_size']).to(device)


def get_unroll_steps(unroll_steps_type: str, labels: torch.Tensor, epoch: int) -> int:
    """
    get number of unroll_steps depending on unroll_steps_type

    :param unroll_steps_type: type from yaml file
    :param labels: y values (ground truth)
    :param epoch: current epoch
    :return: number of steps to unroll the RNN
    """
    if unroll_steps_type == 'full_length':
        return labels.shape[1]
    elif unroll_steps_type == 'batch_length':
        return np.max(np.argwhere(labels.detach().numpy() == 3)[:, 1])
    elif unroll_steps_type == 'batch_number':
        return int(2 + np.ceil(epoch / 2))
    else:
        raise ConfigurationError('Unknown unroll_steps_type.')


if __name__ == '__main__':
    model_name = f'default'

    params = parse_yaml(model_name, 'param')
    print(f'run {model_name} on  {torch.cuda.get_device_name()}')

    batch_size = params['batch_size']
    unroll_steps_type = params.get('unroll_steps_type', 'full_length')  # batch_length, batch_number

    grad_clip = params.get('grad_clip', None)

    embed_pretrained = params.get('embed_pretrained', False)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    if params.get('image_augmentation', False):
        transform_aug = T.Compose([T.Resize(256), T.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.9, 1.2), shear=10), T.RandomPerspective(), T.RandomHorizontalFlip(), T.CenterCrop(224), T.ToTensor(), normalize])
        data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform_aug, max_vocab_size=params['max_vocab_size'], all_lower=params['all_lower'])
    else:
        data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=params['max_vocab_size'], all_lower=params['all_lower'])

    if embed_pretrained:
        pretrained_embeds = PretrainedEmbeddings("embeddings/glove_shrinked.txt", data_train.get_corpus(), device)
    else:
        pretrained_embeds = None

    dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=os.cpu_count())  # set num_workers=0 for debugging

    data_dev = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.devImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=params['max_vocab_size'], all_lower=params['all_lower'])
    data_dev.set_corpus_vocab(data_train.get_corpus_vocab())
    dataloader_dev = DataLoader(data_dev, batch_size, num_workers=os.cpu_count())  # os.cpu_count()

    decoder_type = params.get('decoder_type', 'RecurrentDecoder')
    embeddings, model = setup_model(params, data_train, pretrained_embeds)

    tensorboard = Tensorboard(log_dir=f'runs/{model_name}', device=device)
    tensorboard.add_images_with_ground_truth(data_dev)

    if embed_pretrained:
        criterion = nn.CosineEmbeddingLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=data_train.corpus.vocab.stoi[PAD_TOKEN])

    optimizer = optim.Adam(model.parameters(), lr=float(params['learning_rate']), weight_decay=float(params['weight_decay']))
    last_validation_score = float('-inf')

    start_epoch = 0

    model_path = params.get('load_model', None)
    if model_path:
        state_dicts = torch.load(model_path, map_location=device)
        start_epoch = state_dicts['epoch'] + 1
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

    for epoch in trange(start_epoch, params['n_epochs']):  # loop over the dataset multiple times
        model.train()

        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_train)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            unroll_steps = get_unroll_steps(unroll_steps_type, labels, epoch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _, att_probs, _ = model(inputs, labels,
                                             scheduled_sampling=params['scheduled_sampling'],
                                             batch_no=epoch + i / len(dataloader_train),
                                             k=params['scheduled_sampling_k'],
                                             embeddings=embeddings,
                                             unroll_steps=unroll_steps,
                                             decoder_type=decoder_type)

            if embed_pretrained:
                targets = labels[:, 1:unroll_steps].contiguous()
                loss = criterion(outputs, pretrained_embeds(targets.long()), torch.tensor([1]).float().to(device))
            else:
                targets = labels[:, 1:unroll_steps].contiguous().view(-1)
                loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), targets.long())

            if att_probs is not None:  # only with RecurrentDecoder, TransformerDecoder does not have attention
                loss += 1. * ((1. - att_probs.sum(dim=1)) ** 2).mean()  # Doubly stochastic attention regularization
            loss.backward()
            if grad_clip:
                clip_gradient(optimizer, grad_clip)
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
                unroll_steps = get_unroll_steps(unroll_steps_type, labels, epoch)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs, _, att_probs, _ = model(inputs, labels, unroll_steps=unroll_steps, decoder_type=decoder_type)
                targets = labels[:, 1:unroll_steps].contiguous().view(-1)  # shifted by one because of BOS
                loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), targets.long())
                if att_probs is not None:  # only with RecurrentDecoder, TransformerDecoder does not have attention
                    loss += 1. * ((1. - att_probs.sum(dim=1)) ** 2).mean()  # Doubly stochastic attention regularization
                loss_sum += loss.item()

                for beam_size in range(1, len(bleu_1) + 1):
                    prediction, _ = model.predict(data_dev, inputs, data_dev.max_length, beam_size, decoder_type=decoder_type)
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
            tensorboard.add_predicted_text(global_step, data_dev, model, data_dev.max_length, decoder_type=decoder_type)
            tensorboard.writer.flush()

            # Save model, if score got better
            saved_model = params.get('save_model', 'every')
            if saved_model == 'improvement':
                compared_score = bleu_1[0] / len(dataloader_dev)
                if last_validation_score < compared_score:
                    last_validation_score = compared_score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_sum / len(dataloader_dev),
                    }, f'saved_models/{model_name}-bleu_1-{last_validation_score}.pth')
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_sum / len(dataloader_dev),
                }, f'saved_models/{model_name}-epoch={epoch}.pth')
