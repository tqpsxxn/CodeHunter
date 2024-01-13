import re

import javalang
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import traceback
from ast_bert_data_process_util import get_feature_tokens
import logging
logging.basicConfig(filename='new_data_process.log', level=logging.DEBUG)

parse_error_count = 0

PAD = '[PAD]'

# 定义函数用于对单条Java代码进行分词并编码
def tokenize_and_encode(code, tokenizer, word_max_length):
    if len(code) > word_max_length - 2:
        code = code[:word_max_length - 2]
    if (code == PAD):
        return [0] * word_max_length, [0] * word_max_length
    code = tokenizer.tokenize(code)
    if len(code) > word_max_length - 2:
        code = code[:word_max_length - 2]
    tokens = ['[CLS]'] + code + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = word_max_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    return input_ids, input_mask

# 把每个函数的代码行数处理为line_limit大小
def limit_lines_length(lines, line_limit, word_max_length):
    PAD = '[PAD]'
    # 如果太长则截断
    if(len(lines) >= line_limit):
        return lines[:line_limit]
    # 如果太短则补齐
    res = lines + [PAD] * (line_limit - len(lines))
    return res

# 把每个函数的label数量处理为line_limit
def limit_tags_length(tags, line_limit):
    if(len(tags) >= line_limit):
        return tags[:line_limit]
    res = tags + [0] * (line_limit - len(tags))
    return res

# 对每个函数进行处理，并将其保存到list中
def processed_data(path, tokenizer, word_max_length = 20, line_limit = 50):
    # 加载tokenizer
    processed_data = []
    # 加载训练数据集
    data = pd.read_pickle(path)
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据
    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        code_lines, tags = row['lines'], row['labels']
        try:

            #  ast解析
            code_lines, tags = get_feature_tokens(code_lines, tags)
            lines_count = len(code_lines)
            if(len(code_lines) == 0):
                continue
            # 把lines处理成line_limit长度
            code_lines = limit_lines_length(code_lines, line_limit, word_max_length)
            tags = limit_tags_length(tags, line_limit)
            # 解析代码
            code_lines = [line.strip() for line in code_lines]
            # 变量名替换
            code_lines = replace_line(code_lines)

            func_data = []
            for j, line in enumerate(code_lines):
                code = code_lines[j]
                label = tags[j]
                input_ids, length = tokenize_and_encode(code, tokenizer, word_max_length)
                func_data.append((input_ids, label, length))

            # 将每一行代码的编码以及对应的标签合并到一个list中，并将其保存到一个字典中
            func_data = list(zip(*func_data))
            # 处理成bert模型需要的格式，lines表示每一行代码，labels表示每行代码的标记，lengths表示attention_mask,attention_mask表示一行中哪些单词是需要进行训练的（0）
            # 如果是填充的则不需要用于训练（1）
            func_dict = {
                'lines': func_data[0],
                'labels': func_data[1],
                'lengths': func_data[2],
                'lines_count': lines_count
            }

            # 将每个函数的数据保存到一个list中
            processed_data.append(func_dict)

        except Exception as e:
            # 如果解析代码失败，直接跳过
            print(e)
            continue
    logging.info("processed_data count:{}".format(len(processed_data)))
    return processed_data

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


if __name__ == "__main__":
    # 将所有函数的数据保存到文件中
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    pd.to_pickle(processed_data('data/train.pkl', tokenizer, 64, 50), 'processed_train_ast.pkl')
    pd.to_pickle(processed_data('data/valid.pkl', tokenizer, 64, 50), 'processed_valid_ast.pkl')
    pd.to_pickle(processed_data('data/test.pkl', tokenizer, 64, 50), 'processed_test_ast.pkl')