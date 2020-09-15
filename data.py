import itertools
import os
from collections import defaultdict

from PIL import Image
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN
from joeynmt.vocabulary import Vocabulary
from torch.utils.data import Dataset
from torchtext import data


class Flickr8k(Dataset):

    def __init__(self, data_path: str, split_file_name: str, ann_file_name: str, transform=None, fix_length: int = None, max_vocab_size: int = None):
        """
        Flickr Dataset class to use with dataloader
        :param data_path: dataset directory
        :param split_file_name: file listing all used images in split - vary this parameter for train/test split
        :param ann_file_name: file containing annotation tokens
        :param transform: torchvision transforms object to be applied on the images
        :param fix_length: pads caption fix_length if provided, otherwise pads to the length of the longest example in the batch
        :param max_vocab_size: the maximum size of the vocabulary, or None for no maximum
        """
        self.root = os.path.expanduser(data_path)
        self.ann_file = os.path.expanduser(ann_file_name)
        self.transform = transform

        self.idx2image = []
        self.idx2caption = []
        self.idx2caption_no_padding = []
        self.image_name2idxs = defaultdict(list)
        self.all_captions = []  # Captions on ENTIRE dataset (TRAIN+DEV+TEST)
        self.lengths = dict()

        # Get image file names for chosen TRAIN/DEV/TEST data
        valid_image_file_names = set([line.rstrip() for line in open(split_file_name, 'r')])
        annotations = [line.rstrip() for line in open(ann_file_name, 'r')]

        valid_counter = 0
        for annotation in annotations:
            image_file_name, caption = annotation.split('\t')
            self.all_captions.append(caption.lower().split())
            if image_file_name[:-2] in valid_image_file_names:
                if len(caption.split()) not in self.lengths:
                    self.lengths[len(caption.split())] = [valid_counter]
                else:
                    self.lengths[len(caption.split())].append(valid_counter)
                self.idx2caption_no_padding.append(caption.lower().split())
                self.idx2image.append(image_file_name[:-2])
                self.idx2caption.append(caption.lower().split())
                self.image_name2idxs[image_file_name[:-2]].append(len(self.idx2image) - 1)
                valid_counter += 1

        self.corpus = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, fix_length=fix_length)
        self.corpus.vocab = Vocabulary(tokens=sorted(list(set(itertools.chain(*self.all_captions)))))  # Corpus containing possible tokens across TRAIN+DEV+TEST
        self.idx2caption = self.corpus.pad(self.idx2caption)
        self.max_length = max(list(self.lengths.keys()))

    def __getitem__(self, index):
        image_name = self.idx2image[index]

        # Image
        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions, for image
        caption = self.corpus.numericalize([self.idx2caption[index]]).squeeze()
        return img, caption, image_name

    def get_all_references_for_image_name(self, image_name: str):
        references = []
        for idx in self.image_name2idxs[image_name]:
            references.append(self.corpus.numericalize([self.idx2caption[idx]]).squeeze().detach().numpy().tolist()[1:])
        return references

    def __len__(self):
        return len(self.idx2caption)

