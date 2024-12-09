import os
import pickle
import re
from math import ceil
import random

import torch
from sklearn.model_selection import train_test_split
from tokenizers.decoders import WordPiece
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import BertTokenizerFast

from params import DatasetTypes
from utils import logger

datasets_path = './datasets/'


def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def remove_consecutive_spaces(string):
    return ' '.join(string.split())


def chars(f, t):
    return list(map(chr, range(ord(f), ord(t) + 1)))


valid_chars = set(chars('а', 'я') + ['ё'] + chars('a', 'z') + chars('0', '9')
                  + list("<>(){}" + ".,\"!?;:-*—\'"))
invalid_regex = '[^' + ''.join(
    chars('а', 'я') + ['ё'] + chars('a', 'z') + chars('0', '9') + list(r"<>(){}" + ".,\"!?;:*—\' -")) + ']'


# Приведите текст к нижнему регистру
# yберите лишние символы и пробелы идущие подряд
def clean_text(text):
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub(invalid_regex, "", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    return text


def load_texts(paths):
    texts = []
    for path in paths:
        text = read_file(path)
        text = clean_text(text)
        texts.append(text)
    return texts


def get_all_files(directory, fraction=0.0):
    paths = []
    for path, folders, files in os.walk(directory):
        for filename in files:
            full_path = os.path.join(path, filename)
            # print(full_path)
            if full_path.endswith('.txt'):
                paths.append(full_path)

    if fraction:
        random.shuffle(paths)
        return paths[:int(fraction * len(paths))]
    else:
        return paths


def combine_texts(texts):
    tokens = []
    for text in tqdm(texts):
        tokens += encode(text)
    return tokens


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, max_seq_len, offset):
        logger.info("Start combining...")
        tokens = combine_texts(texts)

        logger.info("Start splitting onto chunks...")
        self.chunks = []
        n_iter = ceil(len(tokens) / (max_seq_len - offset)) + 1
        for i in tqdm(range(n_iter)):
            start_index = i * (max_seq_len - offset)
            if start_index >= len(tokens):
                continue
            end_index = i * (max_seq_len - offset) + max_seq_len
            self.chunks.append(tokens[start_index: end_index])

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)


def collate_batch(batch):
    src_list = []
    for src in batch:
        src_list.append(torch.tensor(src))

    return pad_sequence(src_list, padding_value=pad_token_idx, batch_first=True)


def get_datasets(path_to_texts, max_seq_len=257, offset=0, use_serialized=True, dataset_type=DatasetTypes.whole):
    max_seq_len = 257 if not (dataset_type in [DatasetTypes.half_seq_len, DatasetTypes.half_tiny]) else 129
    if dataset_type == DatasetTypes.small:
        logger.info('Getting small datasets...')
        dataset_filename = 'small_datasets.pkl'
    elif dataset_type == DatasetTypes.tiny:
        logger.info('Getting tiny datasets...')
        dataset_filename = 'tiny_datasets.pkl'
    elif dataset_type == DatasetTypes.whole:
        logger.info('Getting whole datasets...')
        dataset_filename = 'datasets.pkl'
    elif dataset_type == DatasetTypes.half_seq_len:
        logger.info('Getting half seq len datasets...')
        dataset_filename = 'half_seq_len_dataset.pkl'
    elif dataset_type == DatasetTypes.half_tiny:
        logger.info('Getting half seq len tiny datasets...')
        dataset_filename = 'half_seq_len_tiny_dataset.pkl'
    else:
        raise ValueError('Wrong DatasetType!')

    if use_serialized:
        if os.path.exists(datasets_path + dataset_filename):
            with open(datasets_path + dataset_filename, 'rb') as f:
                train_data, test_data = pickle.load(f)
            logger.info(
                "Loaded serialized datasets. Sizes of them are: " + str(len(train_data)) + ', ' + str(len(test_data)))
            return train_data, test_data

    if dataset_type == DatasetTypes.small:
        paths = get_all_files(path_to_texts, fraction=0.15)
    elif dataset_type == DatasetTypes.tiny or dataset_type == DatasetTypes.half_tiny:
        paths = get_all_files(path_to_texts, fraction=0.03)
    else:
        paths = get_all_files(path_to_texts)

    paths.sort()

    paths_train, paths_test = train_test_split(paths, test_size=0.03, random_state=12345)

    texts_train = load_texts(paths_train)
    texts_test = load_texts(paths_test)

    train_data = TextDataset(texts_train, max_seq_len, offset)
    test_data = TextDataset(texts_test, max_seq_len, 0)
    if use_serialized:
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)

        with open(datasets_path + dataset_filename, 'wb') as f:
            pickle.dump((train_data, test_data), f)
        logger.info("Datasets was serialized. Sizes of them are: " + str(len(train_data)) + ", " + str(len(test_data)))
    return train_data, test_data


def get_tokenizer(path='./data/vocab.txt'):
    return BertTokenizerFast(path)


tokenizer = get_tokenizer(path='./data/vocab.txt')
vocab_size = tokenizer.vocab_size
pad_token_idx = 0
sep_token_idx = 3
decoder = WordPiece()


def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)


def decode(sequence):
    return decoder.decode(tokenizer.convert_ids_to_tokens(sequence))
