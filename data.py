import itertools
import os
from collections import defaultdict, Counter
from typing import List, Tuple

import torch
from PIL import Image
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN
from joeynmt.vocabulary import Vocabulary
from torch.utils.data import Dataset
from torchtext import data


class Flickr8k(Dataset):
    """
    Custom Dataset class for the Flickr Dataset
    """

    def __init__(self, data_path: str, split_file_name: str, ann_file_name: str, transform=None, fix_length: int = None, max_vocab_size: int = None, all_lower: bool = False):
        """
        Flickr Dataset class to use with dataloader
        :param data_path: Dataset directory
        :param split_file_name: File listing all used images in split - vary this parameter for train/test split
        :param ann_file_name: File containing annotation tokens
        :param transform: Torchvision transforms object to be applied on the images
        :param fix_length: Pads caption fix_length if provided, otherwise pads to the length of the longest example in the batch
        :param max_vocab_size: The maximum size of the vocabulary, or None for no maximum
        :param all_lower: Set this to convert all tokens to lower case
        """
        self.root = os.path.expanduser(data_path)
        self.ann_file = os.path.expanduser(ann_file_name)
        self.transform = transform
        self.max_vocab_size = max_vocab_size

        self.idx2image = []
        self.idx2caption = []
        self.idx2caption_no_padding = []
        self.image_name2idxs = defaultdict(list)
        self.lengths = dict()

        # Get image file names for chosen TRAIN/DEV/TEST data
        valid_image_file_names = set([line.rstrip() for line in open(split_file_name, 'r')])
        annotations = [line.rstrip() for line in open(ann_file_name, 'r')]

        valid_counter = 0
        # Loop through all annotations, as they are not separated per fraction (train/dev/test).
        for annotation in annotations:
            image_file_name, caption = annotation.split('\t')
            # Only choose the captions for images, which are part of the current fraction defined.
            if image_file_name[:-2] in valid_image_file_names:
                # In case this option is enabled, convert all tokens in lower letters.
                if all_lower:
                    caption = caption.lower().split()
                    self.idx2caption.append(caption)
                else:
                    caption = caption.split()
                    self.idx2caption.append(caption)

                # Store each caption id corresponding a caption length in a dictionary
                # ...this can be used to sample batches of equal size
                if len(caption) not in self.lengths:
                    self.lengths[len(caption)] = [valid_counter]
                else:
                    self.lengths[len(caption)].append(valid_counter)
                self.idx2caption_no_padding.append(caption)
                self.idx2image.append(image_file_name[:-2])
                self.image_name2idxs[image_file_name[:-2]].append(len(self.idx2image) - 1)
                valid_counter += 1

        self.corpus = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, fix_length=fix_length)
        self.max_length = max(list(self.lengths.keys()))

        # Pad captions
        self.idx2caption = self.corpus.pad(self.idx2caption)

        # Select the most-frequently used tokens (top max_vocab_size) and build vocabulary object
        counter = Counter(list(itertools.chain(*self.idx2caption)))
        vocab_tokens = sort_and_cut(counter, self.max_vocab_size)
        self.corpus.vocab = Vocabulary(tokens=vocab_tokens)

    def get_corpus_vocab(self) -> Vocabulary:
        """
        As the vocabulary  used for testing has to be the same as for training, this method makes it possible
        to obtain the training dataset's vocab and use it in the test dataset, as well. The purpose of this is to let
        the dataset mark tokens, which are not included in the train dataset as unknown.
        :return: torchtext.vocab object to be used by another dataset
        """
        return self.corpus.vocab

    def set_corpus_vocab(self, corpus_vocab: Vocabulary) -> None:
        """
        See get_corpus_vocab, this set function allows to change the dataset's vocabulary to the one
        used during training
        :param corpus_vocab: torchtext.vocab object from the train dataset
        """
        self.corpus.vocab = corpus_vocab

    def get_corpus(self):
        return self.corpus

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get data from dataset on index position
        :param index: Index number
        :return: Image as tensor, true captions as tensor of token number, name of the image file
        """
        image_name = self.idx2image[index]

        # Image
        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions, for image
        caption = self.corpus.numericalize([self.idx2caption[index]]).squeeze()
        return img, caption, image_name

    def get_all_references_for_image_name(self, image_name: str):
        """
        There are multiple captions per image. For evaluation purposes, this function returns a list of tensors of token
        numbers of all captions given the image name
        :param image_name: File name of the image
        :return: List of tensors of token numbers
        """
        references = []
        # Loop through all captions related with the given image name, obtain token numbers to replace strings and
        # pack all tensors of token numbers in a list
        for idx in self.image_name2idxs[image_name]:
            references.append(self.corpus.numericalize([self.idx2caption[idx]]).squeeze().detach().numpy().tolist()[1:])
        return references

    def __len__(self) -> int:
        return len(self.idx2caption)


def sort_and_cut(counter: Counter, limit: int) -> List[str]:
    """
    This function returns an list of the most-frequent tokens in descending order with given limit
    :param counter: Counter object to retrieve item and frequency from
    :param limit: Number of tokens to be included
    :return: List of the most-used tokens
    """
    """ Cut counter to most frequent, sorted numerically and alphabetically (copied from joeynmt)"""
    # sort by frequency, then alphabetically
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens
