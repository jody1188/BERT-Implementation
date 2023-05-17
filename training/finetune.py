import os
import sys

import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import BERT, BERTClassifier
from utils import ScheduledOptim, save_checkpoint

import tqdm

import matplotlib.pyplot as plt


class Trainer:
    

    def __init__(self, bert : BERT, n_labels, emb_dim, dropout_prob, epochs, checkpoint_dir, train_dataloader, valid_dataloader,
                                                    learning_rate, betas, weight_decay, warmup_steps, log_freq, steps, device):

        
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.emb_dim = emb_dim
        self.dropout_prob = dropout_prob
        self.bert = bert
        self.n_labels = n_labels

        self.bertcls = BERTClassifier(self.bert, self.emb_dim, self.n_labels, self.dropout_prob).to(self.device)
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = Adam(self.bertcls.parameters(), lr=learning_rate, betas = betas, weight_decay = weight_decay)
        self.optimizer_schedule = ScheduledOptim(self.optimizer, emb_dim, n_warmup_steps = warmup_steps)

        self.criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        self.epochs = epochs
        self.steps = steps

        print("Total Parameters:", sum([p.nelement() for p in self.bertcls.parameters()]))


    def training(self):

            print('-' * 20)

            print("Start Training!")

            save_epoch = 0
            outstanding_model_epoch = 0
            outstanding_model_loss = float('inf')

            train_history=[]
            valid_history=[]

            for epoch in range(self.epochs):
                

                sys.stdout.write(f"######### Epoch : {epoch + 1} / {self.epochs} ###########")

                data_iterator = tqdm.tqdm(enumerate(self.train_dataloader),
                              total=len(self.train_dataloader),
                              bar_format="{l_bar}{r_bar}")


                train_loss_per_epoch = 0.0

                self.bertcls.train()
        
                for batch_idx, batch in data_iterator:
                                     
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    logits = self.bertcls.forward(batch["bert_input"], batch["segment_label"])

                    self.optimizer.zero_grad()

                    loss = self.criterion(logits, batch["label"])

                    train_loss_per_iter = loss.item()
                    train_loss_per_epoch += train_loss_per_iter
                    
                    loss.backward()

                    self.optimizer_schedule.step_and_update_lr()

                    if ((batch_idx + 1) % self.steps == 0) and (batch_idx > 0):                        
                        sys.stdout.write(f"###Training### |  Epoch: {epoch + 1} |  Step: {(batch_idx + 1 / len(self.train_dataloader))} | Loss: {train_loss_per_iter}")
                        sys.stdout.write('\n')

                train_history.append(train_loss_per_epoch / len(self.train_dataloader)

                ##### Evaluation #####git remote add origin
                self.bertcls.eval()

                data_iterator = tqdm.tqdm(enumerate(self.valid_dataloader),
                              total=len(self.valid_dataloader),
                              bar_format="{l_bar}{r_bar}")

                for batch_idx, batch in data_iterator:

                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    logits = self.bertcls.forward(batch["bert_input"], batch["segment_label"])
                    _, label_pred = logits.max(1)
                    result = (label_pred == batch["label"]).float()
                    accuracy = result.mean()

                save_checkpoint(self.bertcls, epoch, self.checkpoint_dir)

                sys.stdout.write(f"Validation |  Epoch: {epoch + 1}  | Accuracy : {accuracy}")
                sys.stdout.write('\n')      


            print("Complete Finetuning BERT!")
           
            self.loss_plot(train_history, valid_history, self.checkpoint_dir)


    def loss_plot(self, train_history, save_path):
        
        epoch_size = np.linspace(0, self.epochs, self.epochs)

        plt.plot(epoch_size, np.array(train_history), label = "Training History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plot_dir = save_path 

        plt.savefig(plot_dir + "loss_plot.png")
        print("Save Plot!")