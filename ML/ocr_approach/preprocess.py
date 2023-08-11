import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from custom_dataset import CustomOCRDataset
from helper import create_data_csv, map_values

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch):
    file = open('./vocab.pkl', 'rb')
    vocab = pickle.load(file)
    file.close()
    
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: map_values(x)
    
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

def preprocess():
    create_data_csv('../data/')
    custom_dataset = CustomOCRDataset('./dataset.csv')
    data_iter = iter(custom_dataset)
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])    
    vocab.set_default_index(vocab["<unk>"])

    dataset = pd.read_csv('./dataset.csv')
    train, test = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], shuffle=True)

    train.to_csv('./train_dataset.csv',index=False)
    test.to_csv('./test_dataset.csv',index=False)

    file = open('./vocab.pkl', 'wb')
    # dump information to that file
    pickle.dump(vocab, file)
    # close the file
    file.close()
    return custom_dataset
