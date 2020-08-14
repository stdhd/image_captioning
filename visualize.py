from datetime import datetime

import matplotlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data import Flickr8k
from model import Image2Caption

matplotlib.use('TkAgg')


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


def sequence_to_text(dataset: Flickr8k, seq: np.array):
    return '    ' + ' '.join([dataset.corpus.vocab.itos[id] for id in seq])


class Tensorboard:
    def __init__(self, log_dir: str = f'runs/image_captioning_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', image_idxs=[42, 1337], device: str = 'cpu'):
        self.writer = SummaryWriter(log_dir)
        self.image_idxs = image_idxs
        self.device = device

    def add_images_with_ground_truth(self, dataset: Flickr8k):
        for image_idx in self.image_idxs:
            img, caption = dataset[image_idx]
            self.writer.add_image(f'image-{image_idx}', normalize_inverse(img).cpu().detach().numpy())
            self.writer.add_text(f'image-{image_idx}', sequence_to_text(dataset, caption), 0)
        self.writer.flush()

    def add_predicted_text(self, global_step: int, dataset: Flickr8k, model: Image2Caption):
        for image_idx in self.image_idxs:
            img, caption = dataset[image_idx]
            img = img.unsqueeze(0).to(self.device)
            caption = caption.unsqueeze(0).to(self.device)
            outputs, _, _, _ = model(img, caption)
            self.writer.add_text(f'image-{image_idx}', sequence_to_text(dataset, torch.argmax(outputs.squeeze(0), dim=-1).cpu().detach()), global_step)
        self.writer.flush()
