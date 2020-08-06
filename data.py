from typing import List

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import csv


class Flickr8k(Dataset):

    def __init__(self, root, split_file, ann_file, transform=None):
        """
        Flickr Dartaset class to use with dataloader
        :param root: dataset directory
        :param split_file: file listing all used images in split - vary this parameter for train/test split
        :param ann_file: file containing annotation tokens
        :param transform: torchvision transforms object to be applied on the images
        """
        self.root = os.path.expanduser(root)
        self.ann_file = os.path.expanduser(ann_file)
        self.transform = transform

        self.word_list = list_of_unique_words(ann_file)
        self.word_to_int = {word: idx for idx, word in enumerate(self.word_list)}
        self.one_hot_matrix = torch.eye(len(self.word_list))

        with open(split_file, 'r') as split_f:
            self.split = [line for line in split_f.readlines()]

        self.annotations = {}
        with open(ann_file, 'r') as f:
            for line in f:
                (key, val) = line.split("	")
                self.annotations[key] = val[:-1].lower().split() # TODO End of seq

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        image_name = self.split[index][:-1]

        # Image
        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions, for each image we have 5 captions
        targets = []
        # for i in range(1):
            # targets.append(self.annotations["{}#{}".format(image_name, i)])
        # targets = [self.one_hot_matrix[self.word_to_int[word]] for word in self.annotations["{}#{}".format(image_name, 0)]]
        targets = [self.one_hot_matrix[self.word_to_int[word]] for word in self.annotations["{}#{}".format(image_name, 0)]]
        target_tensors = torch.Tensor(len(targets), len(self.word_list))
        for i, target in enumerate(targets):
            target_tensors[i] = target
        # targets = torch.Tensor(targets)
        # b = torch.Tensor(len(self.word_list), len(targets))
        # torch.cat(targets, out=b)
        return img, target_tensors

    def __len__(self):
        return len(self.split)


def list_of_unique_words(file_name: str) -> List[str]:
    # TODO combine with Flickr8k (end of seq, start of seq tokens, etc...)
    word_list = []
    with open(file_name) as csv_file:
        data = csv.reader(csv_file, delimiter='\t')
        for row in data:
            word_list.extend(row[-1].lower().split())
    return list(set(word_list))
