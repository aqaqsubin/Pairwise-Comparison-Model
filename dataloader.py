from torch.utils.data import DataLoader, Dataset

from data_utils import read_df
from typing import List

import numpy as np
import textdistance
import warnings

warnings.filterwarnings(action='ignore')

PAD_TOK = '[PAD]'
CLS_TOK = '[CLS]'
SEP_TOK = '[SEP]'

class PCData(Dataset):
    def __init__(self, data_path, tokenizer, cand_size=10, max_len=128, model_type='poly'):
        self._data = read_df(data_path)
        self.cls = CLS_TOK
        self.sep = SEP_TOK
        self.pad = PAD_TOK
        self.max_len = max_len
        self.cand_size = cand_size
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self._data)
    
    def _tokenize(self, sent):
        tokens = self.tokenizer.tokenize(self.cls + str(sent) + self.sep)
        seq_len = len(tokens)
        if seq_len > self.max_len:
            tokens = tokens[:self.max_len-1] + [tokens[-1]]
            seq_len = len(tokens)
            assert seq_len == len(tokens), f'{seq_len} ==? {len(tokens)}'
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return token_ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx].to_dict()
        query = turn['query']
        reply = turn['reply']
        cands = turn['candidates'].tolist()[:self.cand_size]
        label = [1] + [0 for _ in range(self.cand_size)]

        if self.model_type =='cross':
            pair_seq = list(map(lambda x: self._tokenize(query + self.sep + x), [reply] + cands))

            return(pair_seq, label)
        else:
            query_ids = self._tokenize(query)
            cands = list(map(lambda x: self._tokenize(x), [reply] + cands))

        return(query_ids, cands, label)
