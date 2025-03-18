import os
import random
import sys

import torch

import utils
import vocab

random.seed(118)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_dataset():
    x = []
    y = []
    for _ in range(10_000):
        for _x, _y in utils.make_sample():
            x.append(vocab.stoi(_x))
            y.extend(vocab.stoi(_y))
    return torch.tensor(x), torch.tensor(y)


if __name__ == "__main__":
    torch.save(make_dataset(), "data/dataset.pt")
