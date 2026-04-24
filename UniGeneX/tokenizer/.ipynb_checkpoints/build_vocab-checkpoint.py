   
import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
# import torchtext.vocab as torch_vocab
# from torchtext.vocab import Vocab
# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers import AutoTokenizer, BertTokenizer



from .vocab_factory import vocab
from .gene_tokenizer import GeneVocab

def build_vocab(train_genes, save_path):
    print(f' train genes length: {len(list(train_genes))}')
    print(f'will save vocab in {save_path}')
    save_path = Path(save_path)
    iterator = iter(list(train_genes))

    counter = Counter()
    counter.update(iterator)


    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    min_freq = 1
#     word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
    word_vocab = vocab(ordered_dict, min_freq=min_freq)
    gene2idx = word_vocab.get_stoi()

    with open(Path(save_path) / 'default_census_vocab.json', 'w') as outfile:
        json.dump(gene2idx, outfile)


    # vocab_path = '/home/xwanaf/bio/scGPT-dev-temp/scgpt/tokenizer/default_census_vocab.json'
    vocab_path = Path(save_path) / 'default_census_vocab.json'
    with open(vocab_path, 'r') as file:
        vocab_g = json.load(file)

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_value = -2

    vocab_path = save_path / 'default_census_vocab.json'
    vocab_g = GeneVocab.from_file(Path(vocab_path))
    for s in special_tokens:
        if s not in vocab_g:
            vocab_g.append_token(s)

    print(f'vocab length: {len(vocab_g)}')

    