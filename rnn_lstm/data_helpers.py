import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import random

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 正则替换
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r",encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]  # s.strip(rm) 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    negative_examples = list(open(negative_data_file, "r",encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    # print(positive_labels)
    negative_labels = [[1, 0] for _ in negative_examples]
    # print(negative_labels)
    y = np.concatenate([positive_labels, negative_labels], 0)
    print("1=================================")
    # print(x_text)
    print("2=================================")
    # print(y)
    print("3=================================")
    print(len(x_text))
    print(len(y))
    print("4=================================")
    return [x_text, y]

'''
  """
    下面这个是返回的格式
    [
        [
            "hello good ",
            " bad bad "
        ]
        [
            [0,1],
            [1,0]
        ]    
    ]
  """
'''

def load_data_and_labels_QA(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r",encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]  #  s.strip(rm) 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    negative_examples = list(open(negative_data_file, "r",encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    print(positive_labels)
    negative_labels = [[1, 0] for _ in negative_examples]
    print(negative_labels)
    y = np.concatenate([positive_labels, negative_labels], 0)
    print("1=================================")
    print(x_text)
    print("2=================================")
    # print(y)
    print("3=================================")
    print(len(x_text))
    print(len(y))
    print("4=================================")

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def batch_iter2(x,y,num = 100 ):
    x = np.array(x)
    y = np.array(y)
    # random.seed(1)
    # print("------------------------")
    # print(y)
    # print(len(y))
    for step in range(1000):
        s1 = random.randint(0, len(y) - 1)
        s2 = random.randint(0, len(y) - 1)
        # print("s1,s2 : ",s1,s2)
        y[[s1, s2], :] = y[[s2, s1], :]
        x[[s1, s2], :] = x[[s2, s1], :]
    x1 = []
    y2 = []
    num = min(num,len(y)/2)

    for index in range(num):
        x1.append(x[index])
        y2.append(y[index])
    # print(y)
    return  np.array(x1),np.array(y2)

