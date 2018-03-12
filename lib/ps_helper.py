import codecs
# import logging
# import tensorflow as tf
# import gzip
# import json
# import numpy as np
# import os
# import lib.read_utils as read_utils
# import random
# from tensorflow.contrib import learn
# import datetime
# import lib.my_log as mylog
# from lib.config import config
from lib.ct import ct, log_path
# import jieba
import re
from lib.read_utils import TextConverter
# from gensim import models
# from lib.converter.langconv import *
from lib.config import config
import os
import gzip
import gc

from multiprocessing import Pool, Manager
import math
import heapq

from lib.baike_helper import baike_helper


# 同义词集合，
# 1. 【分类标准1】同一个S的 不同P，但是O一样。 记录  P1 = P2 次数
# 2. 【分类标准2】相同句式。相同的非-S句子
# 3. 【分类标准3】答案的类型（数字 时间 其他类型）


class ps_helper:
    # 【分类标准1】同一个S的 不同P，但是O一样。 记录  P1 = P2 次数
    @staticmethod
    def class1(f5='../data/nlpcc2016/5-class/class1.txt',
               f1="../data/nlpcc2016/2-kb/kb-use.v2.txt"):
        bkh = baike_helper()
        bkh.init_spo(f_in=f1)
        keys = bkh.kbqa.keys()
        ps_dict = dict()  # key =  '\t'.join(list(v1)) value =
        for key in keys:
            vs = bkh.kbqa.get(key)
            # if vs[0]
            #  vs[1]
            vs = list(vs)
            vs1 = [x for x in vs[1]]
            if len(ct.clean_str_answer(vs1)) == len(set(ct.clean_str_answer(vs1))):
                # 答案里面没有一样的就跳过
                continue
            _vs_dict = dict()
            # 遍历每个KEY的VS，如果值重复则记录 相同的属性对 到全局里面
            for _vs in vs:
                if _vs[1] in _vs_dict:
                    # if _vs[0] in ps_dict:
                    #     ps_dict[_vs[0]] += 1
                    # else:
                    #     ps_dict[_vs[0]] = 1

                    s1 = _vs_dict[_vs[1]]
                    s1.add(_vs[0])
                    _vs_dict[_vs[1]] = s1
                else:
                    s1 = set()
                    s1.add(_vs[0])
                    _vs_dict[_vs[1]] = s1
            # 去除 属性值- (属性1，属性2) 序列，
            # 将属性1，属性2作为KEY ,次数作为VALue 存到全局
            for (k1, v1) in _vs_dict.items():
                if len(list(v1)) <= 1:
                    continue
                key1 = '\t'.join(list(v1))

                if key1 in ps_dict:
                    ps_dict[key1] += 1
                else:
                    ps_dict[key1] = 1

        tp = ct.sort_dict(ps_dict, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % (t[0], t[1]))
        ct.file_wirte_list(f5, f5s)

        keys = ps_dict.keys()
        words_bag_list = []
        for key in keys:

            words = set(str(key).split('\t'))

            exist = False
            wl_index = -1
            for word in words:  # 遍历每个单词
                for wl_index in range(len(words_bag_list)):  # 这个单词去匹配一遍所有的
                    if word in words_bag_list[wl_index]:
                        exist = True
                        break
                if exist:
                    break
                    # 把当前的words全部整合进去
            if exist:
                for word in words:  # 遍历每个单词
                    words_bag_list[wl_index].add(word)
            else:
                s1 = set()
                for word in words:  # 遍历每个单词
                    s1.add(word)
                words_bag_list.append(s1)
        # 输出 words_bag_list
        f5s = []
        for words_bag in words_bag_list:
            f5s.append('\t'.join(list(words_bag)))
        ct.file_wirte_list(f5 + '.combine.txt', f5s)
        # words_list.append(words)


        # for key in keys:

    ## 【分类表准2】同一个句式不同的P则归位一类，给出重复的次数和 总次数
    @staticmethod
    def class2(f5='../data/nlpcc2016/5-class/class2.txt',
               f1="../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt"):
        f1s = ct.file_read_all_lines_strip(f1)
        f1s_new = [str(x).split('\t')[6] for x in f1s]

        q_patten_set = set()
        q_patten_dict = dict()
        q_count_dict = dict()
        for f1_line in f1s_new:
            q_patten_set.add(f1_line)
        # for q1 in q_patten_set:
        #     q_patten_dict[q1] = set()
        #     q_count_dict[q1] = 0

        gc1 = ct.generate_counter()
        for q1 in q_patten_set:  # 遍历唯一问题集合
            for f1_line in f1s:  # 遍历问题集合
                index = gc1()
                if index % 100000 == 0:
                    print("%d - %d " % (index / 100000, len(q_patten_set) * len(f1s) / 100000))
                _q1 = str(f1_line).split('\t')[6]
                _ps = str(f1_line).split('\t')[3]
                q1 = str(q1)
                if _q1!= '♠' and  _q1.__contains__(q1):  # 相等 或者 包含？
                    if q1 in q_patten_dict:
                        s1 = q_patten_dict[q1]
                        s1.add(_ps)
                        q_patten_dict[q1] = s1
                        q_count_dict[q1] += 1
                    else:
                        s1 = set()
                        s1.add(_ps)
                        try:
                            q_patten_dict[q1] = s1
                        except Exception as e11:
                            print(e11)
                        q_count_dict[q1] = 1

        tp = ct.sort_dict(q_count_dict)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s\t%s" % (t[0], t[1], '\t'.join(list(q_patten_dict[t[0]]))))
        ct.file_wirte_list(f5, f5s)

        #  -------

        keys = q_patten_dict.keys()
        words_bag_list = []
        for key in keys:

            # words = set(str(key).split('\t'))
            words = q_patten_dict.get(key)  # words  规划总面积	建筑面积	显示器尺寸	面积	占地总面积

            exist = False
            wl_index = -1
            for word in words:  # 遍历每个单词
                for wl_index in range(len(words_bag_list)):  # 这个单词去匹配一遍所有的
                    if word in words_bag_list[wl_index]:
                        exist = True
                        break
                if exist:
                    break
                    # 把当前的words全部整合进去
            if exist:
                wbl = words_bag_list[wl_index]
                for word in words:  # 遍历每个单词
                    wbl.add(word)
                words_bag_list[wl_index] = wbl
            else:
                s1 = set()
                for word in words:  # 遍历每个单词
                    s1.add(word)
                words_bag_list.append(s1)
        # 输出 words_bag_list
        f5s = []
        for words_bag in words_bag_list:
            f5s.append('\t'.join(list(words_bag)))
        ct.file_wirte_list(f5 + '.combine.txt', f5s)


if __name__ == "__main__":
    if False:
        ps_helper.class1(f5='../data/nlpcc2016/5-class/class1.v2.txt',
                         f1="../data/nlpcc2016/2-kb/kb-use.v2.txt")
    if True:
        ps_helper.class2(f5='../data/nlpcc2016/5-class/class2.v5.txt',
                         f1="../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter-2.txt")
