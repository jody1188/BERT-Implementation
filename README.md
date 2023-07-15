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
If run pretraining version of BERT, choose configuration file and write down "pretrain" in input.
```
python __main__.py pretrain.json
pretrain
```

### Fintune(Glue Dataset)
If run finetuning version of BERT ,choose configuration file and write down "finetune" in input.
Or if run with gluedataset, write down glue dataset name after write down train method
```
python __main__.py finetune.json
finetune
{Glue Dataset Name}
```
