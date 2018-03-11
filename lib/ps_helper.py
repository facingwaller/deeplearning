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
        ps_dict = dict()
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
            for (k1,v1) in _vs_dict.items():
                if len(list(v1))<=1:
                    continue
                key1 = '\t'.join(list(v1))
                value1 = 1
                if key1 in ps_dict:
                    ps_dict[key1] += 1
                else:
                    ps_dict[key1] = 1


        tp = ct.sort_dict(ps_dict, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % (t[0], t[1]))
        ct.file_wirte_list(f5, f5s)


if __name__ == "__main__":
    ps_helper.class1(f5='../data/nlpcc2016/5-class/class1.txt',
                     f1="../data/nlpcc2016/2-kb/kb-use.v2.txt")
