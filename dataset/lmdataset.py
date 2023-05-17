import os
import tqdm
import random

import numpy as np

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class LMDataset(Dataset):

    def __init__(self, data_dir, seq_len, vocab):

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = Tokenizer(self.seq_len)

        data_paths = [os.path.join(self.data_dir,i) for i in os.listdir(self.data_dir)]


        with open(data_paths[1], "r") as f:
            self.corpus = f.read().split('\n')
            

    def __len__(self):

        return len(self.corpus)

        
    def __getitem__(self, item):
        
        prob = random.random()

        s1 = self.corpus[item].strip()
        s2_idx = item + 1

        nsp_label = 1

        if prob > 0.5:
            nsp_label = 0
            while (s2_idx == item + 1) or (s2_idx == item):
                s2_idx = random.randint(0, len(self.corpus))

        if s2_idx >= len(self.corpus):
            s2_idx = random.randint(0, len(self.corpus))
        
        s2 = self.corpus[s2_idx].strip()

        s1_tokens, s1_label = self.random_masking(s1)
        s2_tokens, s2_label = self.random_masking(s2)

        s1 = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + s1_tokens + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
        s2 = s2_tokens + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]

        s1_label = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)] + s1_label + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]
        s2_label = s2_label + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]

        segment_label = ([0 for _ in range(len(s1))] + [1 for _ in range(len(s2))])[:self.seq_len]

        bert_input = (s1 + s2)[:self.seq_len]
        bert_label = (s1_label + s2_label)[:self.seq_len]

        padding = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "nsp_label": nsp_label}

        return {key: torch.tensor(value) for key, value in output.items()}




    def random_masking(self, sentence):

        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        token_ids = []
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            tmp_token_ids = self.tokenizer.convert_tokens_to_ids(token)

            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    token_ids.append(self.tokenizer.get_mask_token_idx())

                elif prob < 0.9:
                    rand_ids = random.randrange(self.tokenizer.get_vocab_size())
                    token_ids.append(rand_ids)
                else:
                    token_ids.append(tmp_token_ids)
                    
                output_label.append(tmp_token_ids)

            else:
                token_ids.append(tmp_token_ids)
                output_label.append(0)

        return token_ids, output_label


