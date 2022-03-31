import os
from io import open
import re
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


# Vocabulary to store all words and their indices
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def tokenize(dictionary, path, pad_number, lower_char):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Add words to the vocabulary
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                word = word.lower() if lower_char else word
                if pad_number:
                    word = '<num>' if word.isdigit() or isinstance(word, float) else word
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        idss = []
        for line in f:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                word = word.lower() if lower_char else word
                if pad_number:
                    word = '<num>' if word.isdigit() or isinstance(word, float) else word
                ids.append(dictionary.word2idx[word])
            idss += ids

    return dictionary, idss


class WikiTextData(Dataset):
    """ WikiText Dataset """
    def __init__(self, args, tks_file):
        self.n_gram = args.n_gram
        self.tokens_file = tks_file
        self.length = len(tks_file) - args.n_gram

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.tokens_file[index: index+self.n_gram], self.tokens_file[index+self.n_gram]


def collate_fn(insts):
    """ Batch preprocess """
    seq_tokens_batch, tgt_tokens_batch = list(zip(*insts))

    seq_tokens_batch = torch.LongTensor(seq_tokens_batch)
    tgt_tokens_batch = torch.LongTensor(tgt_tokens_batch)
    return seq_tokens_batch, tgt_tokens_batch


def get_dataloader(args):
    """ Get dataloader and dictionary """

    if os.path.exists(args.path_data_processed):
        with open(args.path_data_processed, 'rb') as f:
            mappings = pickle.load(f)

        vocab = mappings['vocab']
        train_data = mappings['train_data']
        valid_data = mappings['valid_data']
        test_data = mappings['test_data']

    else:
        vocab = Vocabulary()
        vocab, train_data = tokenize(vocab, path=os.path.join(args.path_data, 'train.txt'), pad_number=args.pad_number,
                                     lower_char=args.lower_char)
        vocab, valid_data = tokenize(vocab, path=os.path.join(args.path_data, 'valid.txt'), pad_number=args.pad_number,
                                     lower_char=args.lower_char)
        vocab, test_data = tokenize(vocab, path=os.path.join(args.path_data, 'test.txt'), pad_number=args.pad_number,
                                    lower_char=args.lower_char)
        with open(args.path_data_processed, 'wb') as f:
            mappings = {
                'vocab': vocab,
                'train_data': train_data,
                'valid_data': valid_data,
                'test_data': test_data
            }
            pickle.dump(mappings, f)

    train_loader = DataLoader(WikiTextData(args, train_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(WikiTextData(args, valid_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(WikiTextData(args, test_data), batch_size=args.batch_size, num_workers=args.num_worker,
                             collate_fn=collate_fn, shuffle=True)
    return vocab, train_loader, valid_loader, test_loader
