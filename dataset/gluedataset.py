import os

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from .tokenizer import Tokenizer



data_keys = {
    "cola": ("sentence", None),
    "mnli" : ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    }



class GlueDataset1(Dataset):

    def __init__(self, dataset_name, dataset, seq_len):

        super().__init__() 
        
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        
        self.tokenizer = Tokenizer(self.seq_len)


        self.dataset = dataset

        self.data_tuple = data_keys[self.dataset_name]

        self.X_data = str(self.dataset[self.data_tuple[0]])

        self.label = self.dataset["label"]


    def __len__(self):

        return len(self.dataset)
    

    def __getitem__(self, item):
        
        s1_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.X_data[item]))
        label = self.label[item]

        s1 = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + s1_tokens + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]

        bert_input = s1[:self.seq_len]
        segment_label = [0 for _ in range(len(s1))][:self.seq_len]

        padding = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "label": label,
                  "segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}



class GlueDataset2(Dataset):

    def __init__(self, dataset_name, dataset, seq_len):

        super().__init__() 
        
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.tokenizer = Tokenizer(self.seq_len)
        self.dataset = dataset
        self.data_tuple = data_keys[self.dataset_name]

        self.X_data1 = str(self.dataset[self.data_tuple[0]])
        self.X_data2 = str(self.dataset[self.data_tuple[1]])
        self.label = self.dataset["label"]
        
        #if dataset_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
        #    self.dataset["label"] = ("0", "1", "2")



    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, item):
        
        s1_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.X_data1[item]))
        s2_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.X_data2[item]))
        label = self.label[item]

        s1 = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + s1_tokens + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
        s2 = s2_tokens + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]

        bert_input = (s1 + s2)[:self.seq_len]

        segment_label = ([0 for _ in range(len(s1))] + [1 for _ in range(len(s2))])[:self.seq_len]


        padding = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "label": label,
                  "segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}