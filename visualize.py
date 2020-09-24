from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from joeynmt.constants import BOS_TOKEN, EOS_TOKEN
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data import Flickr8k
from model import Image2Caption


class NormalizeInverse(transforms.Normalize):
    '''
    Undoes the normalization and returns the reconstructed images in the input domain.
    copied from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    '''

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_inverse = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Tensorboard:
    def __init__(self, log_dir: str = f'runs/image_captioning_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', image_idxs=[7, 42, 128, 512, 1337], device: str = 'cpu'):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.image_idxs = image_idxs
        self.device = device

    def add_images_with_ground_truth(self, dataset: Flickr8k):
        for image_idx in self.image_idxs:
            img, caption, image_name = dataset[image_idx]
            self.writer.add_image(f'image-{image_idx}', normalize_inverse(img).cpu().detach().numpy())
            self.writer.add_text(f'image-{image_idx}', '    ' + ' | '.join([' '.join(sentence) for sentence in dataset.corpus.vocab.arrays_to_sentences(dataset.get_all_references_for_image_name(image_name))]), -1)
        self.writer.flush()

    def add_predicted_text(self, global_step: int, dataset: Flickr8k, model: Image2Caption, max_output_length: int, beam_size: int = 1, beam_alpha: float = 0.4):
        for image_idx in self.image_idxs:
            img, _, _ = dataset[image_idx]
            img = img.unsqueeze(0).to(self.device)
            prediction, attention_scores = model.predict(dataset, img, max_output_length, beam_size, beam_alpha)
            decoded_prediction = dataset.corpus.vocab.arrays_to_sentences(prediction)[0]
            self.writer.add_text(f'image-{image_idx}', '    ' + ' '.join(decoded_prediction), global_step)
            visualize_attention(img.squeeze(0), decoded_prediction, attention_scores[0], dataset.max_length, f'{self.log_dir}/{image_idx}-step_{global_step:03d}.png')
        self.writer.flush()


def visualize_attention(image: torch.Tensor, word_seq: List[str], attention_scores: np.ndarray, max_length: int, file_name: str):
    image = normalize_inverse(image).cpu().detach().numpy()
    image = (image.transpose((1, 2, 0)) * 225).astype(np.uint8)

    height = int(np.ceil((max_length + 1) / 5.))
    fig, ax = plt.subplots(height, 5, figsize=(height * 3, 15))
    [axi.set_axis_off() for axi in ax.ravel()]  # hide axes for subplots

    ax[0][0].set_title(BOS_TOKEN)
    ax[0][0].imshow(image)

    if len(word_seq) > 0:
        if word_seq[-1] != EOS_TOKEN:
            word_seq.append(EOS_TOKEN)

        extent = [0, image.shape[0], image.shape[1], 0]
        attention_score_shape = [np.round(np.sqrt(attention_scores.shape[-1])).astype(np.uint8)] * 2
        for idx, (word, attention_score) in enumerate(zip(word_seq, attention_scores), start=1):
            current_axis = ax[idx // 5][idx % 5]
            current_axis.set_title(word)
            current_axis.imshow(image)
            current_axis.imshow(attention_score.reshape(attention_score_shape), cmap='hot', interpolation='bilinear', alpha=0.5, extent=extent, origin='upper')

    fig.savefig(file_name)
    plt.close(fig)
