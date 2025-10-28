import tiktoken
import torch
from torch.utils.data import DataLoader

from data_loader import GPTDatasetV1


if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Apple Silicon GPU')
else:
    device = torch.device('cpu')
    print('CPU')


def create_data_loader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
