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
# import jieba


class baike_helper:
    def __init__(self):
        # jieba.set_dictionary('../data/jieba_dict/dict.txt.big')
        # self.stopwordset = set()
        # with open('../data/jieba_dict/stopwords.txt', 'r', encoding='utf-8') as sw:
        #     for line in sw:
        #         self.stopwordset.add(line.strip('\n'))
        print(1)

    # 统计关系的数目并做分析，排序
    @staticmethod
    def relatons_statistics():

        file_name = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb"
        ct.print(file_name)

        d_ditc = dict()
        index = 0
        print("4300")
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                index += 1
                if index % 10000 == 0:
                    print(index / 10000)
                if len(line.strip().split('\t')) != 3:
                    continue
                line = line.strip().split('\t')[1].strip()
                if line not in d_ditc:
                    # lines.add(line)
                    d_ditc[line] = 1
                else:
                    d_ditc[line] += 1
        print("OUT sort")
        r_d_sort2 = sorted(d_ditc.items(), key=lambda d: d[1], reverse=True)

        with codecs.open(file_name + "_clear.txt", mode="w", encoding="utf-8") as f1_writer:
            for l in r_d_sort2:
                f1_writer.write("%s\t%s\n" % (l[0], l[1]))

    #

    # 输入文本，输出分词后的文本
    # type = rdf | questions
    def convert_text_to_seg(self, file_in, file_out, type="rdf"):
        with open(file_out, 'w', encoding='utf-8') as f_out:
            with open(file_in, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if type == "rdf":
                        # 增加操作
                        print(124444)
                    if type == "questions":
                        line = line.split('\t')[0]
                    line = line.strip('\n')
                    words = jieba.cut(line, cut_all=False)

                    words_out = []
                    for word in words:
                        if word not in self.stopwordset:
                            words_out.append(word)
                    f_out.write(' '.join(words_out) + '\n')
        print(321321)

    # 清洗百科KB数据spo
    # 1.去除属性中的空白字符 1
    # 2.去除属性中所有非中文、数字和英文字母的字符
    # 3.将实体和属性中的所有大写外文字符转为小写 1
    # 4.p=o的删除掉 1
    @staticmethod
    def clean_baike_kb():
        file_name = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb"
        file_out_name = file_name + ".out.txt"
        clean_log_path = "../data/nlpcc2016/clean_baike_kb.txt"
        ct.print(file_name)
        d_ditc = dict()
        s_set = set()
        index = 0
        # print("4300")
        new_line_list = []
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                index += 1
                if index % 10000 == 0:
                    print("%d / %d" % (index / 10000, 4300))
                if len(line.strip().split('\t')) != 3:
                    ct.just_log(clean_log_path, line)
                    continue
                # 3 大写变小写
                line = line.strip().lower().strip('\n')
                s = line.split('\t')[0]
                p = line.split('\t')[1]
                o = line.split('\t')[2]
                # 4
                if p == o:
                    ct.just_log(clean_log_path, line)
                    continue
                # 2
                p = p.replace(" ", "").replace("•", "").replace("-", "") \
                    .replace("【", "").replace("】", "") \
                    .replace("[", "").replace("]", "")
                new_line = "%s\t%s\t%s" % (s, p, o)
                # 记录RDF
                # ct.just_log(file_name + "_clean1.txt", new_line)
                # new_line_list.append(new_line)
                # 记录实体
                # ct.just_log(file_name + "_clean1_s.txt", s)
                s_set.add(s)
                # 记录关系
                # ct.just_log(file_name + "_clean1_p.txt", p)
                # 记录object
                # ct.just_log(file_name + "_clean1_o.txt", o)

        print("ok1")
        i = 0
        # with open(file_out_name, mode='w', encoding='utf-8') as f_out:
        #     for l in new_line_list:
        #         i+=1
        #         if i %10000 ==0:
        #             print(i)
        #         f_out.write("%s\n"%(l))
        print(5435354)
        s_word_set = set()
        for s in s_set:
            for s1 in s:
                s_word_set.add(s1)
        with open(file_name + "_clean1_s.txt", mode='w', encoding='utf-8') as f_out:
            for s in s_set:
                # ct.just_log(file_name + "_clean1_s.txt", s)
                f_out.write("%s\n" % (s))
        print("_clean1_s.txt")
        # for s in s_word_set:
        #     ct.just_log(file_name + "_clean1_s_word.txt", s)
        with open(file_name + "_clean1_s_word.txt", mode='w', encoding='utf-8') as f_out:
            for s in s_set:
                # ct.just_log(file_name + "_clean1_s.txt", s)
                f_out.write("%s\n" % (s))

        print("_clean1_s_word.txt")

        print(312321)

    @staticmethod
    def get_ngrams(input, n):
        output = {}  # 构造字典
        for i in range(len(input) - n + 1):
            ngramTemp = " ".join(input[i:i + n])  # .encode('utf-8')
            if ngramTemp not in output:  # 词频统计
                output[ngramTemp] = 0  # 典型的字典操作
            output[ngramTemp] += 1
        return output

    # 重新输出实体-长度，并排序,
    @staticmethod
    def statistics_subject_len():
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
        f_out = f_in + ".statistics_len_and_sort.txt"

        d_dict = dict()
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    d_dict[line.strip('\n').strip()] = len(line.strip('\n').strip())
            tp = ct.sort_dict(d_dict, True)
            for t in tp:
                out.write("%s\t%s\n" % (t[0], t[1]))

        print(5435436)

    def init_ner(self):
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
        f_in2 = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt.statistics_len_and_sort.txt"
        d_dict = dict()
        d_set = set()
        begin = False
        end = False
        ct.print_t()
        # s1
        # with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
        #     for l in read_file:
        #         l = l.replace("\r", "").replace("\n", "").split('\t')[0]
        #         d_set.add(l)
        # ct.print_t()
        # s2
        # new_line = line
        # new_line = []
        # for word in line:
        #     if word in d_set:
        #         new_line.append(word)
        # new_line = "".join(new_line)
        # s3

        # 找出N-gram对应的所有可能的单词逐个匹配
        # 3.1 加载N-GRAM的字典
        # f_in2
        ct.print_t("3.1 加载N-GRAM的字典")
        n_gram_dict = dict()
        with codecs.open(f_in2, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                word = line.replace("\r", "").replace("\n", "").split('\t')[0]
                time = str(line.replace("\r", "").replace("\n", "").split('\t')[1])
                if time in n_gram_dict:
                    tmp = n_gram_dict[time]
                    tmp.append(word)
                    n_gram_dict[time] = tmp
                else:
                    n_gram_dict[time] = [word]
        self.n_gram_dict = n_gram_dict
        ct.print_t("init_ner ok")

    # 建造实体的词汇库
    # ner
    # 1 加载词汇表
    # 2 剔除不在词汇表中的字
    # 3 做N-GRAM 匹配
    # 3.1 加载N-GRAM的字典

    def ner(self, line):
        new_line = line
        # ct.print_t("3.2 匹配")
        cand_entitys = []
        new_line_len = len(new_line)
        find = False

        for i in range(new_line_len):
            index = new_line_len - int(i)
            # print(index)
            all_entitys = baike_helper.get_ngrams(new_line, index)
            for entity in all_entitys:
                v = self.n_gram_dict.get(str(index), "")
                if v == "":
                    continue
                # print(v)
                entity = str(entity).replace(" ", "")
                if entity in v:
                    cand_entitys.append(entity)
                    find = True
                    # break  # 暂时先只找第一个试试看
            # if find:
            #     break

        # print(654354353)
        return cand_entitys

    @staticmethod
    def ner_all_stences():

        print(321321)


def build_and_statistics_vocab():
    f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
    f_out = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s_words_statistics.txt"

    d_dict = dict()
    with codecs.open(f_out, mode="w", encoding="utf-8") as out:
        with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                for w in line.strip('\n'):
                    if w in d_dict:
                        d_dict[w] += 1
                    else:
                        d_dict[w] = 1
        tp = ct.sort_dict(d_dict, True)
        for item in tp:
            out.write("%s\t%s\n" % (item[0], item[1]))


def method_name():
    bk = baike_helper()
    f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt"
    f_out = f_in + "-out.txt"
    bk.convert_text_to_seg(f_in, f_out, type="questions")


if __name__ == '__main__':
    # baike_helper.static_relatons()
    # method_name()
    # baike_helper.statistics_subject_len()
    # build_and_statistics_vocab()
    # s ="《机械设计基础》这本书的作者是谁"
    bkh = baike_helper()
    bkh.init_ner()
    f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt"
    index = 0
    with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
        for line in read_file:
            index += 1
            print(index)
            s = line.replace("\r", "").replace("\n", "").split("\t")[0]
            ss = bkh.ner(s)
            if len(ss) > 0:
                ct.just_log("../data/nlpcc2016/extract_entitys.txt", '\t'.join(ss))
            else:
                ct.just_log("../data/nlpcc2016/extract_entitys.txt", "NULL")
            print(ss)
