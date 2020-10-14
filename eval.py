import os

import torch
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from torchvision import transforms as T
from tqdm import tqdm

from data import Flickr8k
from train import setup_model
from yaml_parser import parse_yaml

if __name__ == '__main__':
    """
    This methid can be used to evaluate a trained model on the test split giving BLEU scores 1-4 
    while applying Beam Search with beam sizes 1-4.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = 'weight_decay_scheduled'
    params = parse_yaml(model_name, 'param')

    print(f'run {model_name} on {torch.cuda.get_device_name()}')

    batch_size = params.get('batch_size', 16)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_train = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.trainImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=params['max_vocab_size'], all_lower=params['all_lower'])

    data_test = Flickr8k('data/Flicker8k_Dataset', 'data/Flickr_8k.testImages.txt', 'data/Flickr8k.token.txt', transform=transform, max_vocab_size=params['max_vocab_size'], all_lower=params['all_lower'])
    data_test.set_corpus_vocab(data_train.get_corpus_vocab())
    dataloader_test = DataLoader(data_test, batch_size, num_workers=os.cpu_count())

    embeddings, model = setup_model(params, data_test)

    model_path = params.get('load_model', None)
    if model_path:
        state_dicts = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dicts['model_state_dict'])
    else:
        raise ValueError('Missing `load_model` value.')

    with torch.no_grad():
        model.eval()

        loss_sum = 0
        bleu_1 = [0]
        bleu_2 = [0]
        bleu_3 = [0]
        bleu_4 = [0]
        for data in tqdm(dataloader_test):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, image_names = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            for beam_size in range(1, len(bleu_1) + 1):
                prediction, _ = model.predict(data_test, inputs, data_test.max_length, beam_size)
                decoded_prediction = data_test.corpus.vocab.arrays_to_sentences(prediction)

                # Create ground truth references for each tested image
                decoded_references = []
                for image_name in image_names:
                    decoded_references.append(data_test.corpus.vocab.arrays_to_sentences(data_test.get_all_references_for_image_name(image_name)))

                idx = beam_size - 1

                # Bleu scores for all beam sizes are summed up, such that the average can be calculated at the end
                bleu_1[idx] += bleu_score(decoded_prediction, decoded_references, max_n=1, weights=[1])
                bleu_2[idx] += bleu_score(decoded_prediction, decoded_references, max_n=2, weights=[0.5] * 2)
                bleu_3[idx] += bleu_score(decoded_prediction, decoded_references, max_n=3, weights=[1 / 3] * 3)
                bleu_4[idx] += bleu_score(decoded_prediction, decoded_references, max_n=4, weights=[0.25] * 4)

        # Print averages BLEU scores for each beam size
        for idx in range(len(bleu_1)):
            print(f'BEAM-{idx + 1}/BLEU-1', bleu_1[idx] / len(dataloader_test))
            print(f'BEAM-{idx + 1}/BLEU-2', bleu_2[idx] / len(dataloader_test))
            print(f'BEAM-{idx + 1}/BLEU-3', bleu_3[idx] / len(dataloader_test))
            print(f'BEAM-{idx + 1}/BLEU-4', bleu_4[idx] / len(dataloader_test))
