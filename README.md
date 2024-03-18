# Task1
## Introduction
Used to locate code that may have exceptions, i.e. try block localization

## How to run?
1. Download the compressed file task1_data.zip from [https://pan.quark.cn/s/59b3149262e7](https://pan.quark.cn/s/59b3149262e7) (password:VT7r)and unzip it to the task1/data directory
2. Run `python ast_bert_data_process.py` to preprocess data
3. Run `python bert_lstm_train_new.py train` to train model
4. Run `python bert_lstm_train_new.py test 40` to test model


# Task2
## Introduction
Used to predict the types of exceptions that may arise from possible abnormal code lines


## How to run?
1. Download the compressed file task2_data.zip from [https://pan.quark.cn/s/60aa93a5b00c](https://pan.quark.cn/s/60aa93a5b00c) (password:knuH)and unzip it to the task2/data directory
2. Run `python prepare_bert.py` to preprocess data
3. Run `python bert_train_new.py train` to train model
4. Run `python bert_train_new.py test_topn 40` to test model


# plugin
## Introduction
users can use the functions provided by the IDE plugin to detect and handle exceptions in the program code automatically.

## How to run?
1. move the model parameter files for task1 and task2 to the plugin/checkpoints directory
2. Run `python plugin-web.py` to start the plugin back-end service 
3. Refer to [https://github.com/tqpsxxn/CodeHunterPlugin](https://github.com/tqpsxxn/CodeHunterPlugin) to start the plugin front-end service