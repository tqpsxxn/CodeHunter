# Task1
## Introduction
Used to locate code that may have exceptions, i.e. try block localization

## How to run?
1. Download the compressed file task1_data.zip from [https://pan.quark.cn/s/faa9be169a2b](https://pan.quark.cn/s/faa9be169a2b) and unzip it to the task1/data directory
2. run `python ast_bert_data_process.py` to preprocess data
3. run `python bert_lstm_train_new.py train` to train model
4. run `python bert_lstm_train_new.py test 40` to test model

# Task2
## Introduction
Used to predict the types of exceptions that may arise from possible abnormal code lines

## How to run?
1. Download the compressed file task2_data.zip from [https://pan.quark.cn/s/cf7647ede57c](https://pan.quark.cn/s/cf7647ede57c) and unzip it to the task2/data directory
2. run `python prepare_bert.py` to preprocess data
3. run `python bert_train_new.py train` to train model
4. run `python bert_train_new.py test_topn 40` to test model
