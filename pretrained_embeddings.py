import torch
import torch.nn as nn
import numpy


class PretrainedEmbeddings(nn.Module):
    def __init__(self, file_path, corpus):
        super().__init__()
        self.token2embedding_tensor = torch.empty(10_000, 300)
        with open(file_path) as file:
            lines = file.read().splitlines()
        for idx, line in enumerate(lines):
            values = line.split()
            word = values[0]
            embedding = torch.from_numpy(numpy.array(values[1:], dtype=float))
            token_id = corpus.numericalize([[word]]).item()
            self.token2embedding_tensor[token_id] = embedding

    def forward(self, x):
        result = torch.index_select(self.token2embedding_tensor, 0, x.flatten())
        if len(list(x.size())) > 1:
            result = result.reshape(x.size(0), x.size(1), 300)
        return result


