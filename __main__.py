import os
import json
import tqdm

from utils import arg_parse
from collections import namedtuple

from tokenizers import ByteLevelBPETokenizer

import torch
from torch.utils.data import DataLoader
from dataset import LMDataset, GlueDataset1, GlueDataset2

from datasets import load_dataset

from model import BERT

from training import Pretrainer
from training import Trainer


    ######################## Pretrain ########################

def main(train_mode):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_mode == "pretrain":

        print("####### Start Pretraining BERT! #######")

        args = arg_parse()
        
        with open(args.cfgs, 'r') as f: 
            cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

        print("####### Preparing Pretrain Dataset #######")

        pretrain_dataset = LMDataset(cfgs.data_dir, tokenizer, cfgs.seq_len, vocab)

        dataset_len = len(pretrain_dataset)
    
        train_dataset_len = int(dataset_len * (1 - cfgs.valid_ratio))
        valid_dataset_len = dataset_len - train_dataset_len


        train_dataset, valid_dataset = torch.utils.data.random_split(pretrain_dataset, 
                                                            [train_dataset_len, valid_dataset_len])



        train_dataloader = DataLoader(train_dataset, batch_size = cfgs.batch_size, num_workers = cfgs.num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size = cfgs.batch_size, num_workers = cfgs.num_workers)

        print("####### Finish Prepare DataLoader! #######")

        bert = BERT(cfgs.vocab_size, cfgs.seq_len, cfgs.emb_dim, cfgs.ff_dim, cfgs.n_layers, cfgs.n_heads, cfgs.dropout_prob, device)

        print("Finsh!")

        pretrainer = Pretrainer(bert, cfgs.emb_dim, cfgs.vocab_size, cfgs.epochs, cfgs.save_epoch, cfgs.checkpoint_dir, train_dataloader, valid_dataloader, 
                                                    cfgs.learning_rate, (cfgs.adam_beta1, cfgs.adam_beta2), cfgs.weight_decay, cfgs.warmup_steps, cfgs.log_freq, cfgs.steps, device)

        print("####### Start Training! #######")

        pretrainer.training()

        print("####### End Training! #######")



    ######################## Finetune ########################

    elif train_mode == "finetune":

        print("####### Start Finetuning BERT! #######")

        args = arg_parse()
        
        with open(args.cfgs, 'r') as f: 
            cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

        single_sentence = ["cola", "sst2"]
        double_sentence = ["stsb", "rte", "wnli", "qqp", "ax", "mrpc", "mnli_matched", "mnli_mismatched"]

        dataset_name = input("Enter Dataset : ")


        ############### For GLUE Dataset ###############    

        if dataset_name in ["mnli_matched", "mnli_mismatched"]:
            train_dataset = load_dataset("glue", "mnli", split = "train")
            valid_dataset = load_dataset("glue", dataset_name, split = "validation")
        else:
            train_dataset = load_dataset("glue", dataset_name, split = "train")
            valid_dataset = load_dataset("glue", dataset_name, split = "validation")

        if dataset_name in double_sentence:
            train_dataset = GlueDataset2(dataset_name, train_dataset, cfgs.seq_len)
            valid_dataset = GlueDataset2(dataset_name, valid_dataset, cfgs.seq_len)
        
        else:
            train_dataset = GlueDataset1(dataset_name, train_dataset, cfgs.seq_len)
            valid_dataset = GlueDataset1(dataset_name, valid_dataset, cfgs.seq_len)


        print("####### Finish Prepare Dataset! #######")


        train_dataloader = DataLoader(train_dataset, batch_size = cfgs.batch_size, num_workers = cfgs.num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size = cfgs.batch_size, num_workers = cfgs.num_workers)

        print("####### Finish Prepare DataLoader! #######")


        bert = BERT(cfgs.vocab_size, cfgs.seq_len, cfgs.emb_dim, cfgs.ff_dim, cfgs.n_layers, cfgs.n_heads, cfgs.dropout_prob, device)

        betas = (cfgs.adam_beta1, cfgs.adam_beta2)

        finetuner = Trainer(bert, cfgs.n_classes, cfgs.emb_dim, cfgs.dropout_prob, cfgs.epochs, cfgs.checkpoint_dir, train_dataloader, valid_dataloader,
                                                    cfgs.learning_rate, betas, cfgs.weight_decay, cfgs.warmup_steps, cfgs.log_freq, cfgs.steps, device)

        print("####### Start Training! #######")

        finetuner.finetuning()

        print("####### End Training! #######")



if __name__ == "__main__":

    train_mode = input("Enter Training Mode : ")

    main(train_mode)