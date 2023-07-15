# BERT

This repository is implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)] in Pytorch.

## Requirements

First, install requirements.
```
pip install -r requirements.txt
```

## Data preparation

Then, prepare your own (pretrain or finetune)data in file/data.

## Usage

### Pretraining
If implement pretraining version of BERT, choose configuration file and write down "pretrain".
```
python __main__.py pretrain.json
```
