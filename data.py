import os

from PIL import Image
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN
from torch.utils.data import Dataset
from torchtext import data


class Flickr8k(Dataset):

    def __init__(self, data_path: str, split_file_name: str, ann_file_name: str, transform=None, fix_length: int = None):
        """
        Flickr Dataset class to use with dataloader
        :param data_path: dataset directory
        :param split_file_name: file listing all used images in split - vary this parameter for train/test split
        :param ann_file_name: file containing annotation tokens
        :param transform: torchvision transforms object to be applied on the images
        :param fix_length: pads caption fix_length if provided, otherwise pads to the length of the longest example in the batch
        """
        self.root = os.path.expanduser(data_path)
        self.ann_file = os.path.expanduser(ann_file_name)
        self.transform = transform

        self.idx2image = []
        self.idx2caption = []
        self.all_captions = []  # Captions on ENTIRE dataset (TRAIN+DEV+TEST)

        # Get image file names for chosen TRAIN/DEV/TEST data
        valid_image_file_names = set([line.rstrip() for line in open(split_file_name, 'r')])
        annotations = [line.rstrip() for line in open(ann_file_name, 'r')]

        for annotation in annotations:
            image_file_name, caption = annotation.split('\t')
            self.all_captions.append(caption.lower().split())
            if image_file_name[:-2] in valid_image_file_names:
                self.idx2image.append(image_file_name[:-2])
                self.idx2caption.append(caption.lower().split())

        self.corpus = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, fix_length=fix_length)
        self.corpus.build_vocab(self.all_captions)  # Corpus containing possible tokens across TRAIN+DEV+TEST
        self.idx2caption = self.corpus.pad(self.idx2caption)

    def __getitem__(self, index):
        image_name = self.idx2image[index]

        # Image
        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions, for image
        caption = self.corpus.numericalize([self.idx2caption[index]]).squeeze()
        return img, caption

    def __len__(self):
        return len(self.idx2caption)
