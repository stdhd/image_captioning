from data import Flickr8k
from torchvision import transforms as T

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
data_train = Flickr8k('Flicker8k_Dataset', 'Flickr_8k.trainImages.txt', 'Flickr8k.token.txt', transform=transform)
