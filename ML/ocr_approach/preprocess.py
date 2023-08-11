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
    """Token Generator

    Parameters
    ----------
    data_iter : iter
        Data iteration object

    Yields
    ------
    int
        the tokenized number for the text
    """
    # Iterate over the iter object and tokenize the text in each index and yield it
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch):
    """Collate function, function that processes the text with different length with ease

    Parameters
    ----------
    batch : DataLoader Batch
        the batch of text being processed

    Returns
    -------
    Dataloader Object with offset
        Offset is calculated and appended on each return
    """
    # Read the vocabulary from a file
    file = open('./vocab.pkl', 'rb')
    vocab = pickle.load(file)
    file.close()
    
    # Defining Pipeline for text: Tokenize the input text
    text_pipeline = lambda x: vocab(tokenizer(x))
    # Defining Mapping of the labels to acceptable range
    label_pipeline = lambda x: map_values(x)
    
    # Apply the pipeline process in each item in a Dataloader batch
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        # Lenght of the processed text
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) # Cumulative Sum of offsets of all the items
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

def preprocess():
    """The first function that runs creates the necessary csv, and builds the vocabulary

    Returns
    -------
    CustomOCRDataset
        the train split of the dataset
    """
    # Create initial dataset csv
    create_data_csv('../data/')

    # Create the Custom Dataset from the csv
    custom_dataset = CustomOCRDataset('./dataset.csv')
    data_iter = iter(custom_dataset)

    # Build the vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])    
    vocab.set_default_index(vocab["<unk>"])

    # Split the data to test and train
    dataset = pd.read_csv('./dataset.csv')
    train, test = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], shuffle=True)
    train.to_csv('./train_dataset.csv',index=False)
    test.to_csv('./test_dataset.csv',index=False)

    # Dump the vocabulary in pickle file
    file = open('./vocab.pkl', 'wb')
    pickle.dump(vocab, file)
    file.close()

    # Create CustomDataset for training and return
    custom_dataset = CustomOCRDataset('./train_dataset.csv')
    return custom_dataset
