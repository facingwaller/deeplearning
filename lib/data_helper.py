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

mylog.logger.info("test")


# 从文件中读取问题集合
# 返回句子和标签



class free_base:
    entitys = []

    def init_fb(self, file_name="../data/freebase/freebase_entity.txt"):
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.entitys.append(line.replace("\r\n", ""))

    def find_fb_by_id(self, id):
        exist = False

        for e1 in self.entitys:

            if str(e1) == str(id):
                exist = True
                print(e1, id)
        return exist


def read_fb(entity_name):
    file_name = "../data/freebase/freebase_entity.txt"
    return


# 从文件中压缩包中读取指定的entity
def find_fb_by_id2(entity_name):
    print("-->entity_name")
    file_name = "../data/freebase/freebase_entity.txt"
    print(file_name)
    lines = []
    with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        for line in read_file.readlines():
            lines.append(line)

    return


def test1():
    # read_files("../data/sq/annotated_fb_data_train-1.txt")
    # read_fb("1111.json")

    fb1 = free_base()
    fb1.init_fb()
    print(fb1.entitys.__len__())
    r = fb1.find_fb_by_id("012_0k9")
    print(r)

    return


# =======================================================================simple questions
def test2():
    d = DataClass()
    print(d.batch_iter(2))


def read_file(file_name):
    """
    读取文件返回行的list
    :param file_name:
    :return:
    """
    idx = 0
    lines = []
    with codecs.open(file_name, mode="r", encoding="utf-8") as file:
        try:
            for line in file.readlines():
                idx += 1
                lines.append(line)
        except Exception as e:
            print("index = ", idx)
            logging.error("error ", e)
    return lines


class DataClass:
    # ---------------------freebase
    entitys = []
    # ---------------------DataClass
    train_path = "../data/simple_questions/annotated_fb_data_train-1.txt"
    entity1_list = []  # id
    entity1_value_list = []  # 值

    relation_list = []  # 单词版
    entity2_list = []
    question_list = []
    question_list_index = []  # 数字索引版
    relation_list_index = []

    train_question_list_index = []  # 数字索引版
    train_relation_list_index = []
    test_question_list_index = []  # 数字索引版
    test_relation_list_index = []

    fb = []

    # ----------------------分割数据，预处理，转换成index形式

    def get_split_list(self, sentence_list):
        """
        将句子列表的所有空格隔开的单词全部取出来
        :param sentence:
        :return:
        """
        q_words = []
        for q in sentence_list:
            # q = str(q).replace("\n\r", " ")
            q_words_list = q.split(" ")
            for word in q_words_list:
                q_words.append(word)
        return q_words

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
            q_words_list = sentence.split(" ")
            for word in q_words_list:
                one_sentence.append(word)
            all_stence.append(one_sentence)
        return all_stence

    def __init__(self):
        self.init_simple_questions()
        self.init_fb()
        print("init finish!")
        # 将问题和关系的字符串变成以空格隔开的一个单词的list

        # total_list = self.question_list + self.relation_list
        q_words = self.get_split_list(self.question_list)
        q_words.extend(self.get_split_list(self.relation_list))

        self.converter = read_utils.TextConverter(q_words)
        self.converter.save_to_file_raw(
            "../data/vocab/" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + str(".txt"))
        # self.converter.save_to_file("model/converter.pkl")
        # print(self.converter)

        # 将问题/关系转换成index的系列表示
        self.max_document_length = max([len(x.split(" ")) for x in self.question_list])  # 获取单行的最大的长度
        # 预处理问题和关系使得他们的长度的固定的？LSTM应该不需要固定长度？

        self.question_list_split = self.get_split_list_per_line(self.question_list)
        self.relation_list_split = self.get_split_list_per_line(self.relation_list)
        for q_l_s in self.question_list_split:
            self.question_list_index.append(self.converter.text_to_arr_list(q_l_s))
        # self.relation_list_index = self.converter.text_to_arr(self.relation_list_split)
        for _ in self.relation_list_split:
            self.relation_list_index.append(self.converter.text_to_arr_list(_))
        # 第一版本先padding到max长度
        for s in self.question_list_index:
            padding = self.max_document_length - len(s)
            for index in range(padding):
                s.append(self.max_document_length - 1)  # 用最后一个单词 补齐
            s = np.array(s)
        for s in self.relation_list_index:
            padding = self.max_document_length - len(s)
            for index in range(padding):
                s.append(self.max_document_length - 1)  # 用最后一个单词 补齐
            s = np.array(s)
            # print(1)
        # 按比例分割训练和测试集
        rate = 0.8
        self.train_question_list_index, self.test_question_list_index = \
            self.cap_nums(self.question_list_index, rate)
        self.train_relation_list_index, self.test_relation_list_index = \
            self.cap_nums(self.relation_list_index, rate)
        print(1)

    # ---------------------load_all_train_data
    def load_all_train_data(self):
        """

        :return:返回问题集合,答案集合（关系集合）
        """
        all_data = []

        return self.question_list, self.relation_list

    # ---------------------web questions
    def init_simple_questions(self):
        """
        TODO: 需要给出entity1_list的具体text，暂时不需要
        :return:
        """
        self.entity1_list, self.relation_list, self.entity2_list, self.question_list = \
            self.read_annotated_fb_data_train(self.train_path)

    def read_annotated_fb_data_train(self, file_name):
        line_list = []
        idx = 0
        entity1_list = []
        relation_list = []
        entity2_list = []
        question_list = []

        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file.readlines():
                    idx += 1
                    line_seg = line.split('\t')
                    # www.freebase.com/m/04whkz5
                    entity1 = line_seg[0].split('/')[2]
                    relation1 = line_seg[1].replace("www.freebase.com/", "").replace("/", "_").replace("_", " ")
                    entity2 = line_seg[2].split('/')[2]
                    question = line_seg[3].replace("\r\n", "")

                    entity1_list.append(entity1)
                    relation_list.append(relation1)
                    entity2_list.append(entity2)
                    question_list.append(question)
                    # check it
                    line_list.append(line)
            except Exception as e:
                print("index = ", idx)
                logging.error("error ", e)
        logging.info("load embedding finish!")
        return entity1_list, relation_list, entity2_list, question_list

    # ---------------------embedding
    def embedding(self, input_data, VOCAB_SIZE, HIDDEN_SIZE):
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        # 原本的batch_size*num_steps个单词ID
        # 转为单词向量，转化后输入层维度是batch_size*num_steps*hidden_size
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        return inputs

    # ---------------------freebase
    def init_fb(self, file_name="../data/freebase/freebase_entity.txt"):
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.entitys.append(line.replace("\r\n", ""))

    def exist_in_fb_by_id(self, id):
        """
        是否存在freebase中
        :param id:
        :return:
        """
        exist = False
        for e1 in self.entitys:
            if str(e1) == str(id):
                exist = True
                print(e1, id)
        return exist

    # --------------------生成batch
    def batch_iter(self, question_list_index,relation_list_index,batch_size=100):
        """
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        length = len(x)
        shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        # print("shuffle_indices", str(shuffle_indices))
        total = 0
        for index in shuffle_indices:
            x_new.append(x[index])
            y_new.append(y[index])
            total += 1
            if total >= batch_size:
                break
        # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
        # rate = 1
        r_si = reversed(shuffle_indices)
        r_si = list(r_si)
        # print(r_si)
        total = 0
        for index in r_si:
            z_new.append(y[index])
            total += 1
            if total >= batch_size:
                break
        print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --------------------按比例分割
    def cap_nums(self, y, rate=0.8):
        y = y.copy()
        y = np.array(y)
        s = 0
        total_len = len(y)
        total_index = total_len * rate + 1
        e = int(total_index)

        reverseIndex = int(total_len - total_index)
        # print(reverseIndex)
        # 正向截取
        # 逆向
        y1 = y[s:e]  # [ > s and <= e  ]
        # print(type(y1))

        y2 = y[:reverseIndex]

        # print(total_len)
        print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
        return y1, y2


if __name__ == "__main__":
    test2()
