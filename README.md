# Implementation of PrLM
This is the official implementation of the paper "PrLM: Learning Explicit Reasoning for Personalized RAG via
Contrastive Reward Optimization" based on PyTorch.

## Satisfy the requirements
You need to check it according to the requirements file.

```
According to https://github.com/huggingface/open-r1
```

## Quick Start

### Model preparation

The DeepSeek-R1-Distill-Qwen-1.5B, bge-base-en-v1.5 and BERT can be downloaded from [huggingface](https://huggingface.co/models).

### Data preparation.

Download [LaMP](https://lamp-benchmark.github.io/download) data.

Then process the data:

```bash
cd Personal_RAG
python data/preprocess_profile.py --data_phase train
python data/preprocess_profile.py --data_phase dev
python data/get_user_set.py
```

### Retrieve 

```bash
cd Personal_RAG

python ranking.py --rank_stage retrieval --data_split train --ret_type dense --base_retriever_path base_retriever_path

python ranking.py --rank_stage retrieval --data_split dev --ret_type dense --base_retriever_path base_retriever_path
```

### Contrastive Reward Model Training

Build training data
```bash
cd Personal_RAG
# for positive samples
python ranking.py --rank_stage retrieval --data_split train --ret_type dense --base_retriever_path base_retriever_path

python generation/generate.py --source Personal_RAG/qwen1.5b_outputs/LaMP_7_time/train/recency/bge-base-en-v1.5_5/retrieval --file_name base

# for negtive samples
python ranking.py --rank_stage retrieval --data_split train --ret_type zero_shot

python generation/generate.py --source Personal_RAG/qwen1.5b_outputs/LaMP_7_time/train/recency/zero_shot/retrieval --file_name base
```

Train BERT
```bash
cd Model_Train/Personal_RM

# modify the corresponding file path
python train_bert.py
```

### PrLM RL train

Create dataset

```bash
cd Model_Train

python datasets/create_dataset.py 
```

Training

```bash
cd Model_Train

# for LaMP-7
nohup bash train_model/train_lamp7.sh > train_lamp7.log 2>&1 &

# for LaMP-5
nohup bash train_model/train_lamp5.sh > train_lamp5.log 2>&1 &

# for LaMP-4
nohup bash train_model/train_lamp4.sh > train_lamp4.log 2>&1 &
```

Check folder `Model_Train` for details.


### Evaluate

Testing

```bash
# first retrieve
python ranking.py --rank_stage retrieval --data_split dev --ret_type dense --base_retriever_path base_retriever_path

# generate
python generation/generate.py --source Personal_RAG/qwen1.5b_outputs/LaMP_7_time/train/recency/bge-base-en-v1.5_5/retrieval  --file_name base
```

Check `Personal_RAG` for details.


## Reference
The PrLM is built based on the following project:
- [openr1](https://github.com/huggingface/open-r1)


## Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.4
* torch version: 2.2.0
* OS: Ubuntu 18.04.5 LTS
* GPU: NVIDIA Geforce RTX A6000
* CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz