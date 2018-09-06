# coding=utf-8
import codecs
import logging
import tensorflow as tf
import gzip
import json
import numpy as np
import os
import lib.read_utils as read_utils
import random
from tensorflow.contrib import learn
import datetime
import lib.my_log as mylog
from lib.config import config
from lib.ct import ct, log_path
from lib.baike_helper import baike_helper

# from gensim import models

# ======================================================================common
import math
import random


class ner_helper:
    def __init__(self):
        self.init_data()
        ct.print("init_cc_questions finish.")
        self.converter = read_utils.TextConverter(filename=config.par('cc_vocab'), type="zh-cn")


        # self.load_all_q_r_tuple(config.get_static_q_num_debug(), config.get_static_num_debug(), is_record=True)
        self.get_max_length()
        # self.q_r_2_arrary_and_padding()
        # # 按比例分割训练和测试集
        # self.division_data(0.8, config.cc_par('real_split_train_test'))
        # self.build_embedding_weight(config.wiki_vector_path(mode))

    # 装载数据
    def init_data(self,f1=''):
        self.question_list = ct.file_read_all_lines_strip(f1)
        # 转换为 数字
        print(1)

    def get_max_length(self):
        max_document_length1 = max([len(x) for x in self.question_list])  # 获取单行的最大的长度
        self.max_document_length = max_document_length1
        # ct.print("q:%s r:%s" % (max_document_length1, max_document_length2), "debug")

    def get_split_list_per_line(self, sentence_list):
        """
        将句子列表中的空格隔开的字符串改成以list形式
        :param sentence_list:
        :return:
        """
        all_stence = []
        for sentence in sentence_list:
            one_sentence = []
            # q = str(q).replace("\n\r", " ")
            # if self.mode=='cc':
            #     q_words_list = [x for x in sentence]
            # else:
            q_words_list = sentence.split(" ")
            for word in q_words_list:
                one_sentence.append(word)
            all_stence.append(one_sentence)
        return all_stence

    def q_r_2_arrary_and_padding(self):
        self.question_list_index = []
        # 把问题和关系变成array形式
        self.question_list_split = self.get_split_list_per_line(self.question_list)

        for q_l_s in self.question_list_split:
            self.question_list_index.append(self.converter.text_to_arr_list(q_l_s))

        # 第一版本先padding到max长度
        padding_num = self.get_padding_num()
        for index in range(0, len(self.question_list_index)):
            self.question_list_index[index] = \
                ct.padding_line(self.question_list_index[index], self.max_document_length, padding_num)

    # ---------------------------------零碎的小东西
    def get_padding_num(self):
        return self.converter.vocab_size - 1