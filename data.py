from torch.utils.data import Dataset
import os
from PIL import Image
import math

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
        with open(split_file, 'r') as split_f:
            self.split = [line for line in split_f.readlines()]

        self.annotations = {}
        with open(ann_file, 'r') as f:
            for line in f:
                (key, val) = line.split("	")
                self.annotations[key] = val[:-1]

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        image_name = self.split[index][:-1]

        # Image
        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions, for each image we have 5 captions
        targets = []
        for i in range(5):
            targets.append(self.annotations["{}#{}".format(image_name, i)])

        return img, targets

    def __len__(self):
        return len(self.split)