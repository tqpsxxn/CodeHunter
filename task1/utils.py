# coding=utf-8
import torch
import torchtext
from collections import Counter, OrderedDict

from tqdm import tqdm
import pandas as pd
import os
# from collections import Counter, OrderedDict
import javalang

import re

from ast_bert_data_process_util import get_feature_tokens
import logging
logging.basicConfig(filename='new_data_utils.log', level=logging.DEBUG)

def camel_case_split(string):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)

#读取数据集，平铺读取
def  read_pkl_span_for_bert(pkl_folder, split, line_limit, word_limit):
    assert split in {'train', 'test'}
    print(os.path.join(pkl_folder, split + '.pkl'))
    data = pd.read_pickle(os.path.join(pkl_folder, split + '.pkl'))
    lines_span = list()
    tags_span = list()

    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        lines, tags = row['lines'], row['labels']
        lines = replace_line(lines)
        for j, (s, t) in enumerate(zip(lines, tags)):
            lines_span.append(s)
            tags_span.append(t)
    return lines_span, tags_span

# 从原始的pkl数据集中读取数据
def  read_pkl(pkl_folder, split, line_limit, word_limit):
    assert split in {'train', 'test'}
    docs = []
    labels = []
    lens = list()
    word_counter = Counter()
    print(os.path.join(pkl_folder, split + '.pkl'))
    data = pd.read_pickle(os.path.join(pkl_folder, split + '.pkl'))
    index = 0
    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        if(i > 100000):
            break
        lines, tags = row['lines'], row['labels']
        lines, tags = get_feature_tokens(lines, tags)
        lines = replace_line(lines)
        if(len(lines) == 0):
            continue


        words = list()
        annotations = list()
        assert len(lines) == len(tags)
        # 对lines和tags做遍历，s代表一行代码，t代表对应标识
        for j, (s, t) in enumerate(zip(lines, tags)):
            if j == line_limit:
                break


            w = s.split()
            lens.append(len(w))
            w = w[:word_limit]



            if len(w) == 0:
                print("Invalid line:", s)
            else:

                words.append(w)
                word_counter.update(w)
                annotations.append(t)

        # If all lines were empty
        if len(words) == 0:
            continue

        # 在数据预处理的时候记录被处理的源代码及每一行的正确标记（0 1）
        if index < 1000:
            with open('words.txt', "a") as sourceCodeFile:
                sourceCodeFile.write(str(words).replace('\n', '\t\t\t\t') + '\n')
            with open('annotations.txt', "a") as annotationsFile:
                annotationsFile.write(str(annotations) + '\n')
            index = index + 1


        labels.append(annotations)
        docs.append(words)

    logging.info("MAX LONG WORDS  COUNT: %d", max(lens))
    logging.info("WORDS LENGTH > 32 COUNT: %d", len(list(filter(lambda x: x > 32, lens))))
    logging.info("WORDS LENGTH > 20 COUNT: %d", len(list(filter(lambda x: x > 20, lens))))
    logging.info("WORDS LENGTH > 40 COUNT: %d", len(list(filter(lambda x: x > 40, lens))))
    logging.info("WORDS LENGTH > 50 COUNT: %d", len(list(filter(lambda x: x > 50, lens))))
    logging.info("WORDS LENGTH > 60 COUNT: %d", len(list(filter(lambda x: x > 60, lens))))
    return docs, labels, word_counter


# 处理一行代码，匿名化对象名，将对象名全部与类名进行关联
def replace_line(lines):
    #处理前： line = '         ByteBuffer   writeBuf   =   ByteBuffer . allocateDirect ( CAPACITY_NORMAL ) ;'
    #处理后： line = 'ByteBuffer   byteBuffer   =   ByteBuffer . allocateDirect ( CAPACITY_NORMAL ) ;'
    new_lines = []
    hit_words = {}
    for i in lines:
        temp = i.lstrip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
        for k,v in hit_words.items():
            temp.replace(k, v)
        i_words = temp.lstrip().split(' ')
        #第三个字符为=或；判断为对象定义（存在疑问点：该对象名可能会被作为局部变量使用，但不代表该类的对象）
        if len(i_words) > 2 and (i_words[2] == '=' or i_words[2] == ';'):
            replacedWord = replaceHeadBig(i_words[0])
            hit_words[i_words[1]] = replacedWord
            i_words[1] = replacedWord
            temp = " ".join(i_words)
        new_lines.append(temp)
    return new_lines


#开头字母转小写
def replaceHeadBig(word):
    if word == '' or word == None:
        return word
    return word[0].lower()+word[1:]

def test():
    str = '         ByteBuffer   writeBuf   =   ByteBuffer . allocateDirect ( CAPACITY_NORMAL ) ;'
    lines  = []

def create_input_files(pkl_folder, output_folder, line_limit, word_limit, min_word_count=5, vocab_size=50000):
    # Read training data
    print('\nReading and preprocessing training data...\n')
    os.makedirs(output_folder, exist_ok=True)
    # word_counter=每个单词出现的次数
    train_docs, train_labels, word_counter = read_pkl(pkl_folder, 'train', line_limit, word_limit)

    # Word2Vec().load('output/word2vec.model')
    # 构造词汇表，小于5次则丢弃，不在词汇表中则会被匿名化成unk
    vocab = torchtext.vocab.Vocab(word_counter, max_size=vocab_size, min_freq=min_word_count)
    # vocab = torchtext.vocab.Vocab(word_counter)

    # sorted_by_freq_tuples = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    # ordered_dict = OrderedDict(sorted_by_freq_tuples)
    # vocab = torchtext.vocab.Vocab(ordered_dict)
    # vocab.set_default_index(-1)

    print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
        min_word_count, len(vocab.stoi)))

    tmaps = {"0": 0, "1": 1, "<pad>": 2, "<start>": 3, "<end>": 4}

    # 将词汇表保存到vocab.pt中
    torch.save(vocab, os.path.join(output_folder, 'vocab.pt'))
    print('Vocabulary saved to %s.\n' % os.path.join(output_folder, 'vocab.pt'))
    #得到<pad>和<unk>的索引
    PAD, UNK = vocab.stoi['<pad>'], vocab.stoi['<unk>']
    # Encode and pad
    print('Encoding and padding training data...\n')
    # 对文档内容进行编码
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: vocab.stoi.get(w, UNK), s)) +
            [PAD] * (word_limit - len(s)), doc)) +
                        [[PAD] * word_limit] * (line_limit - len(doc)), train_docs))
    # 每一个文档有几行代码
    # [
    #     {
    #         "dockey": 1
    #     }
    # ]
    lines_per_train_document = list(map(lambda doc: len(doc), train_docs))
    # 每一行代码有几个单词
    # [
    #     {
    #         "dockey": [
    #             {
    #                 "line": 1
    #             }
    #         ]
    #     }
    # ]
    words_per_train_line = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (line_limit - len(doc)), train_docs))
    train_labels = list(
        map(lambda y: y + [tmaps['<pad>']] * (line_limit - len(y)), train_labels))
    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(lines_per_train_document) == len(
        words_per_train_line)
    # Because of the large data, saving as a JSON can be very slow
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'lines_per_document': lines_per_train_document,
                'words_per_line': words_per_train_line},
               os.path.join(output_folder, 'PART_TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to %s.\n' % os.path.abspath(output_folder))

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, lines_per_train_document, words_per_train_line

    # Read test data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_pkl(pkl_folder, 'test', line_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: vocab.stoi.get(w, UNK), s)) +
            [PAD] * (word_limit - len(s)), doc)) +
                        [[PAD] * word_limit] * (line_limit - len(doc)), test_docs))
    lines_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_line = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (line_limit - len(doc)), test_docs))
    test_labels = list(
        map(lambda y: y + [2] * (line_limit - len(y)), test_labels))

    # i = 0
    # for lines in encoded_test_docs:
    #     file_util.writeCode2File('encoded_test_docs.txt', i, str(lines), True)
    #     i = i+1
    #
    # i = 0
    # for lines in lines_per_test_document:
    #     file_util.writeCode2File('lines_per_test_document.txt', i, str(lines), True)
    #     i = i + 1
    #
    # i = 0
    # for lines in words_per_test_line:
    #     file_util.writeCode2File('words_per_test_line.txt', i, str(lines), True)
    #     i = i + 1
    #
    # i = 0
    # for lines in test_labels:
    #     file_util.writeCode2File('test_labels.txt', i, str(lines), True)
    #     i = i + 1


    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(lines_per_test_document) == len(
        words_per_test_line)
    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'lines_per_document': lines_per_test_document,
                'words_per_line': words_per_test_line},
               os.path.join(output_folder, 'PART_TEST_data.pth.tar'))
    print('Encoded, padded test data saved to %s.\n' % os.path.abspath(output_folder))

def save_checkpoint(epoch, model, optimizer, word_map):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map}
    filename = 'ast_checkpoints/checkpoint_%d.pth.tar' % epoch
    torch.save(state, filename)

def ast_nexgen_save_checkpoint(epoch, model, optimizer, word_map):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map}
    filename = 'ast_nexgen_checkpoints/checkpoint_%d.pth.tar' % epoch
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

##tqp
def get_source_file_content():
    train_docs, train_labels, word_counter = read_pkl('./data', 'test', 50, 20)
    # print(train_docs)

if __name__ == '__main__':
    # get_source_file_content()
    # rewriteTestFile(pkl_folder='./data',
    #                    output_folder='./output',
    #                    line_limit=50,
    #                    word_limit=20,
    #                    min_word_count=0)
    # todo 1、当前词典（vacab）大小50000，实际未超出50000，可考虑调小一点提升运行效率
    # done 2、之前的策略是对词典之外的单词进行匿名化处理，当前有一个优化策略已实际应用：识别对象名，将千变万化的对象名统一修改为"首字母小写之后的类名"
    #    例如ByteBuffer   writeBuf   =   ByteBuffer . allocateDirect ( CAPACITY_NORMAL )
    #    现在修改了源代码，会writeBuf自动修改为byteBuffer,在后续byteBuffer被调用时可以一定程度上保留相关信息


    create_input_files(pkl_folder='./data',
                       output_folder='./output',
                       line_limit=50,
                       word_limit=20,
                       min_word_count=0)