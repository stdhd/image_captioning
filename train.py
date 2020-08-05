from data import Flickr8k
from torchvision import transforms as T
transform = T.Compose([T.Resize(256), T.CenterCrop(224)]) , T.ToTensor()])
dat = Flickr8k('Flicker8k_Dataset', 'Flickr_8k.trainImages.txt', 'Flickr8k.token.txt', transform=transform)

