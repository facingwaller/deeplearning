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

MAX_POOL_NUM = 5


class baike_helper:
    # 统计关系的数目并做分析，排序
    @staticmethod
    def relatons_statistics(f1="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb",
                            f2="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt"):
        d_ditc = dict()
        index = 0
        with codecs.open(f1, mode="r", encoding="utf-8") as read_file:
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

        with codecs.open(f2, mode="w", encoding="utf-8") as f1_writer:
            for l in r_d_sort2:
                f1_writer.write("%s\t%s\n" % (l[0], l[1]))

    # F2.0 清洗百科KB数据spo
    # 1.去除属性中的空白字符 1
    # 2.去除属性中所有非中文、数字和英文字母的字符
    # 3.将实体和属性中的所有大写外文字符转为小写 1
    # 4.p=o的删除掉 1
    @staticmethod
    def clean_baike_kb(file_name="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb",
                       file_out_name="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt",
                       clean_log_path="../data/nlpcc2016/clean_baike_kb.txt"
                       ):

        ct.print(file_name)
        d_ditc = dict()
        s_set = set()
        index = 0
        p_set = set()
        o_set = set()

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
                line = line.strip().lower().strip('\n').strip('\r')
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
                new_line_list.append(new_line)
                # 记录实体
                # ct.just_log(file_name + "_clean1_s.txt", s)
                # s_set.add(s)
                # 记录关系
                # ct.just_log(file_name + "_clean1_p.txt", p)
                # 记录object
                # ct.just_log(file_name + "_clean1_o.txt", o)
                # o_set.add(o)

        print("ok1")
        i = 0
        # with open(file_out_name, mode='w', encoding='utf-8') as f_out:
        #     for l in new_line_list:
        #         i+=1
        #         if i %10000 ==0:
        #             print(i)
        #         f_out.write("%s\n"%(l))
        print(5435354)
        index = 0
        with open(file_out_name, mode='w', encoding='utf-8') as f_out:
            for s in new_line_list:
                index += 1
                if index % 10000 == 0:
                    print("%d / %d" % (index / 10000, 4300))
                    f_out.write("%s\n" % (s))
        print("_clean1_s.txt")

        # s_word_set = set()
        # for s in s_set:
        #     for s1 in s:
        #         s_word_set.add(s1)
        # with open(file_name + "_clean1_s.txt", mode='w', encoding='utf-8') as f_out:
        #     for s in s_set:
        #         # ct.just_log(file_name + "_clean1_s.txt", s)
        #         f_out.write("%s\n" % (s))
        # print("_clean1_s.txt")
        # for s in s_word_set:
        #     ct.just_log(file_name + "_clean1_s_word.txt", s)
        # with open(file_name + "_clean1_s_word.txt", mode='w', encoding='utf-8') as f_out:
        #     for s in s_word_set:
        #         # ct.just_log(file_name + "_clean1_s.txt", s)
        #         f_out.write("%s\n" % (s))

        # print("_clean1_s_word.txt")
        # with open(file_name + "_clean1_o.txt", mode='w', encoding='utf-8') as f_out:
        #     for s in o_set:
        #         # ct.just_log(file_name + "_clean1_s.txt", s)
        #         f_out.write("%s\n" % (s))

        print(312321)

    #
    def extract_kb(self,
                   f1='../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt',
                   f2="../data/nlpcc2016/demo1/kb.txt",
                   f3='../data/nlpcc2016/demo1/q.rdf.txt'):
        f3s = ct.file_read_all_lines_strip(f3)
        print(len(f3s))
        # f3s = [str(x).split('\t')[2].lower() for x in f3s]
        f3s_new = []
        for x in f3s:
            x1 = str(x).split('\t')
            if len(x1) < 3:
                print(x)
                continue
            f3s_new.append(x1[2].lower())
        f3s = f3s_new

        print(f3s[0])
        print(f3s[1])

        self.init_spo(f_in=f1)

        with open(f2, mode='w', encoding='utf-8') as o1:
            for f3s_e in f3s:
                vs = self.kbqa.get(f3s_e, "")
                if vs == '':
                    ct.just_log('../data/nlpcc2016/demo1/extract_kb.log.txt', f3s_e)
                    print(f3s_e)
                    continue
                for po in vs:
                    msg = "%s\t%s\t%s" % (f3s_e, po[0], po[1])
                    o1.write(msg + '\n')

        print('ok')

    @staticmethod
    def get_ngrams(input, n):
        output = {}  # 构造字典
        for i in range(len(input) - n + 1):
            ngramTemp = "".join(input[i:i + n])  # .encode('utf-8')
            if ngramTemp not in output:  # 词频统计
                output[ngramTemp] = 0  # 典型的字典操作
            output[ngramTemp] += 1
        return output

    # F2.4 重新输出实体-长度，并排序,
    @staticmethod
    def statistics_subject_len(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
                               ,
                               f_out="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt.statistics_len_and_sort.txt"):

        d_dict = dict()
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    d_dict[line.strip('\n').strip()] = len(line.strip('\n').strip())
            tp = ct.sort_dict(d_dict, True)
            for t in tp:
                out.write("%s\t%s\n" % (t[0], t[1]))

        print(5435436)

    # 合并和重新排序实体
    @staticmethod
    def statistics_and_sort_subject_by_len(f_in, f_out):

        d_dict = dict()
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    line = line.strip('\n').strip()
                    for word in line.split('\t'):
                        d_dict[word.strip('\n').strip()] = len(word.strip('\n').strip())
            tp = ct.sort_dict(d_dict, True)
            for t in tp:
                out.write("%s\t%s\n" % (t[0], t[1]))

    def init_ner(self, f_in2="../data/nlpcc2016/result/e12.txt.statistics.txt"):
        # f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"

        d_dict = dict()
        d_set = set()
        begin = False
        end = False
        #   ct.print_t()
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

    def ner_one(self, i):
        # for i in range(new_line_len):
        index = self.new_line_len - int(i)
        print(index)
        print(self.new_line)
        all_entitys = self.get_ngrams(self.new_line, index)
        for entity in all_entitys:
            v = self.n_gram_dict.get(str(index), "")
            if v == "":
                continue
            # print(v)
            entity = str(entity)
            if entity in v:
                self.cand_entitys.append(entity)
        print(self.cand_entitys)
        # print(123)

    # 建造实体的词汇库
    # ner
    # 1 加载词汇表
    # 2 剔除不在词汇表中的字
    # 3 做N-GRAM 匹配
    # 3.1 加载N-GRAM的字典
    def ner(self, line):
        new_line = line
        # ct.print_t("3.2 匹配")
        self.cand_entitys = []
        new_line_len = len(new_line)
        find = False
        self.new_line = new_line
        self.new_line_len = new_line_len
        allow_more_thread = False
        # ct.print_t('11111111  %d' % new_line_len)
        if allow_more_thread:
            pool = Pool(MAX_POOL_NUM)
            pool.map(self.ner_one, range(0, new_line_len))
            pool.close()
            pool.join()
        else:
            for i in range(new_line_len):
                index = new_line_len - int(i)
                # print(index)
                all_entitys = baike_helper.get_ngrams(new_line, index)
                v = self.n_gram_dict.get(str(index), "")
                if v == "":
                    continue
                for entity in all_entitys:
                    # print(v)
                    entity = str(entity)
                    if entity in v:
                        self.cand_entitys.append(entity)
                        find = True
                        # break  # 暂时先只找第一个试试看
                        # if find:
                        #     break
        # ct.print_t('2222222')
        # print(654354353)
        return self.cand_entitys

    # 不用
    # 构造别名词典
    # 别名词典构建
    # a.以“名”结尾：别名、中文名、英文名、原名等。（第X名、排名等除外,包含 ’第‘ 和 ’排‘ 盛名 的排除）
    # b.以“称”结尾：别称、全称、简称、旧称等。（XX    职称等除外）
    # c.以“名称”结尾：中文名称、其它名称等。（专辑名称、粉丝名称 等除外）
    # 除此之外，如果实体名中存在括号，如“红楼梦（中国古典长篇小说）”，则将括号之外的部分作为
    # 该实体的别名，即“红楼梦”作为实体“红楼梦（中国古典长篇小说）”的别名。如果实体名中包含书名
    # 号，如“《计算机基础》”，则将书名号内的部分作为该实体的别名，即“计算机基础”作为实体“《计
    # 算机基础》”的别名。根据上述方法，最终得到一个包含    7, 304, 663    个别名的别名词典。
    # -------------
    # 输入关系集合，得到关系的别名字典
    # dict - set : 把别名写成一行;以第一个作为字典的key
    # 别名、中文名、英文名、原名、网名
    @staticmethod
    def build_relations_alias_dictory():
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb"
        f_out = f_in + ".alias_dict.txt"
        d_dict = dict()
        w_l = ['别名', '中文名', '英文名', '原名', '二名法',
               '别称', '全称', '简称', '旧称', '中文名称', '其它名称', '外文名']
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    l = ct.clean_str_rn(line).split('\t')[0]
                    if ct.end_with(l, w_l):  # 包含上述则抽取出来
                        if l in d_dict:
                            print(321312)
                            #         d_dict[line.strip('\n').strip()] = len(line.strip('\n').strip())
                            # tp = ct.sort_dict(d_dict, True)
                            # for t in tp:
                            #     out.write("%s\t%s\n" % (t[0], t[1]))

    # 输入关系集合，输出清楚格式的关系名集合;关系别名
    @staticmethod
    def clear_relations(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt",
                        f_out="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt.alais_relations_1.txt"
                        ):

        w_l = ['名', '称']
        w_2 = ['排', '第', '盛名', '专辑名称', '粉丝名称', '签名']

        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    l = ct.clean_str_rel(str(line).split('\t')[0])
                    if ct.end_with(l, w_l):
                        if not ct.contains_with(l, w_2):
                            out.write("%s\n" % l)

    # 抽取正则表达式有误
    # 实体重新抽取   ；实体别名
    # 红楼梦（中国古典长篇小说）
    # 《计算机基础》”，则将书名号内的部分作为该实体的别名
    # 如果抽取出的是唯一的，则是真正的别名
    # @staticmethod
    # def entity_re_extract():
    #     f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt.statistics_len_and_sort.txt"
    #     f_out = "../data/nlpcc2016/result/e_by_m2.txt"
    #
    #     p1 = '(\([^\((]*\))'
    #     p2 = '《[^《]*》'
    #     with codecs.open(f_out, mode="w", encoding="utf-8") as out:
    #         with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
    #             for line in read_file:
    #                 l = line.split('\t')[0]
    #                 r1 = re.findall(p1, l)
    #                 r2 = re.findall(p2, l)
    #
    #                 if len(r1) > 0 or len(r2) > 0:
    #                     list1 = list()
    #                     list1.append(l)
    #                     # out.write("%s\t" % str(l))
    #                     # 第一版本
    #                     # for r in r1:
    #                     #     list1.append(r)
    #                     #     out.write("%s\t" % str(r).strip('《').strip('》')
    #                     #               .strip('(').strip(')'))
    #                     # for r in r2:
    #                     #     list1.append(r)
    #                     #     out.write("%s\t" % str(r).strip('《').strip('》')
    #                     #               .strip('(').strip(')'))
    #                     #
    #                     # 第二版本
    #                     #  张龙(古典名著《三侠五义》中包公的四大侍卫之一)	古典名著《三侠五义》中包公的四大侍卫之一	三侠五义
    #                     r1.extend(r2)
    #                     for r in r1:
    #                         len(r)
    #                     r1.sort(key=lambda x: len(x), reverse=True)
    #                     max_one = r1[0]  # 取出最长的 古典名著《三侠五义》中包公的四大侍卫之一
    #                     l1 = l.replace(max_one, '')  # 张龙 张龙(古典名著《三侠五义》中包公的四大侍卫之一)
    #                     out.write("%s\t" % str(l1))
    #                     out.write("%s\t" % str(l))
    #                     out.write("\n")
    #                     #
    #
    #     print(3421423)
    #
    # @staticmethod
    # # 如果抽取后是空的则不抽取     # 抽取正则表达式有误
    # def entity_re_extract_one(l):
    #     p1 = '(\([^\((]*\))'
    #     p2 = '《[^《]*》'
    #     r1 = re.findall(p1, l)
    #     r2 = re.findall(p2, l)
    #     l1 = l
    #
    #     if len(r1) > 0 or len(r2) > 0:
    #         list1 = list()
    #         list1.append(l)
    #         # 第二版本
    #         #  张龙(古典名著《三侠五义》中包公的四大侍卫之一) 古典名著《三侠五义》中包公的四大侍卫之一    三侠五义
    #         r1.extend(r2)
    #         for r in r1:
    #             len(r)
    #         r1.sort(key=lambda x: len(x), reverse=True)
    #         max_one = r1[0]  # 取出最长的 古典名著《三侠五义》中包公的四大侍卫之一
    #         l1 = l.replace(max_one, '')  # 张龙 张龙(古典名著《三侠五义》中包公的四大侍卫之一)
    #         if l1 == '':  # 如果替换后是空的就不替换
    #             l1 = l
    #     return l1
    #
    #     # 合并 关系对应的实体，将实体别名 和 实体的关系合并起来 并且去重复

    @staticmethod
    # 如果抽取后是空的则不抽取
    # 去掉最后抽取得到的《》或者()
    def entity_re_extract_one_repeat(l):
        # p1 = '(\([^\((]*\))'
        #  p2 = '《[^《]*》'
        p1 = '(\([^\)^\((]*\))'
        p2 = '《[^《^》]*》'
        r1 = re.findall(p1, l)
        r2 = re.findall(p2, l)
        l1 = l

        if len(r1) > 0 or len(r2) > 0:
            list1 = list()
            list1.append(l)
            # 第二版本
            #  张龙(古典名著《三侠五义》中包公的四大侍卫之一) 古典名著《三侠五义》中包公的四大侍卫之一    三侠五义
            r1.extend(r2)
            # for r in r1:
            #     len(r)
            r1.sort(key=lambda x: len(x), reverse=True)
            # 《因为我爱你》(推理小说) 取出括号外面的作为括号里面的别名
            max_one = r1[0]  # 取出最长的 古典名著《三侠五义》中包公的四大侍卫之一
            for r1_1 in r1:
                if r1_1.__contains__('('):
                    max_one = r1_1
                    break
            # print("max_one:"+max_one)

            l1 = l.replace(max_one, '')  # 张龙 张龙(古典名著《三侠五义》中包公的四大侍卫之一)
            if l1 == '':  # 如果替换后是空的就不替换
                l1 = l[1:len(l) - 1]
            if len(re.findall(p1, l1)) or len(re.findall(p2, l1)) > 0:
                l1 = baike_helper.entity_re_extract_one_repeat(l1)
        return l1

        # 合并 关系对应的实体，将实体别名 和 实体的关系合并起来 并且去重复

    # 名字 别名1 别名2 ...
    @staticmethod
    def r_combine():
        # 读取所有的别名
        f1 = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt.alais_relations_1.txt'
        lines = ct.file_read_all_lines(f1)
        all_relations = [str(x).strip('\n') for x in lines]  # 关系集合

        d_dict = dict()
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt"
        index = 0
        with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                index += 1
                if index % 10000 == 0:
                    print(index / 10000)
                ls = str(line).strip('\n').strip('\r').split('\t')
                s = ls[0]
                p = ct.clean_str_rel(ls[1])
                o = ls[2]
                # 首先把自己加入其中
                if s not in d_dict:
                    s1 = set()
                    s1.add(s)
                    d_dict[s] = s1
                #
                if p in all_relations:
                    tmp_set = d_dict[s]
                    if o not in tmp_set:
                        tmp_set.add(o)
                        d_dict[s] = tmp_set
        # 遍历输出
        f_out = "../data/nlpcc2016/result/e_by_m1_e_first.txt"
        print(f_out)
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            for k, v in d_dict.items():
                tmp = []
                tmp.append(k)
                for v1 in v:
                    if v1 not in tmp:
                        tmp.append(v1)
                for v1 in tmp:
                    out.write(v1 + '\t')
                out.write('\n')

        print(4321312)

    # 将实体和关系合并
    # 1 读取实体 和实体的 对应 list
    # 2 读取关系产生的实体和实体的对应，  list[list]
    # 3 用1去循环2，如果1和2相同，则2扩展1
    # 4 输出 这个地方有问题，暂时使用前面2个文件做实体的索引
    @staticmethod
    def e_r_combine():
        entitys = ct.file_read_all_lines_strip(
            '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt.statistics_len_and_sort.txt.alias_dict1-1.txt')
        entitys_r = ct.file_read_all_lines_strip(
            '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt.alais_relations_1.txt.out_e_r_combine-1.txt')
        f_out = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out_e_r_combine-1.txt'
        entitys_r_list1 = dict()
        entitys_r_list2 = dict()
        index = 0
        # 以字典重构 K-V,减少遍历
        for x in entitys_r:
            index += 1
            xs = str(x).split('\t')
            key = xs[0]
            for a in xs:
                if a in entitys_r_list1:
                    # print("%s" % a)  # 都加进去
                    s1 = entitys_r_list1[a]
                    s1.add(a)
                    entitys_r_list1[a] = s1
                else:
                    s1 = set()
                    s1.add(key)
                    entitys_r_list1[a] = s1  # !!!
        print(5555)
        for x in entitys_r:
            index += 1
            xs = str(x).split('\t')
            key = xs[0]
            s1 = set()
            for a in xs:  # 1行
                if a in entitys_r_list1:
                    s1.add(a)
            entitys_r_list2[key] = s1  # !!! 这里将值作为value

        ct.print_t("匹配")
        for x in entitys:
            xs = str(x).split('\t')
            exist = False
            exist_time = 0
            # vv2 = ""
            vv1 = ""
            key = ""
            for xs1 in xs:
                # 遍历，并找到这个key
                vv1 = entitys_r_list1.get(xs1, "")
                if vv1 == "":
                    continue
                # vv2 = vv1
                key = xs1
                exist = True
                # exist_time +=1
                break
                # 假设不存在多处
            # if exist_time>1:
            #     print(xs)
            if exist:
                for vv1_1 in vv1:
                    vv2 = entitys_r_list2.get(vv1_1)
                    for xs1 in xs:
                        vv2.add(xs1)
                    entitys_r_list2[vv1_1] = vv2
        print(321312)
        print(f_out)
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            for k, v in entitys_r_list2.items():
                for v1 in v:
                    out.write(v1 + '\t')
                out.write('\n')

    @staticmethod
    def combine_all_entitys(files, out):
        s1 = set()
        for f in files:
            list1 = ct.file_read_all_lines_strip(f)
            for l in list1:
                for l1 in str(l).split('\t'):
                    s1.add(l1)
        with codecs.open(out, mode="w", encoding="utf-8") as out:
            for w in s1:
                out.write(w + '\n')

        print(4213421)

    @staticmethod
    def build_and_statistics_vocab(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
                                   , f_out="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s_words_statistics.txt"):

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

    def init_find_entity(self):
        self.list1 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e12.txt')
        print('e_by_m1_2')
        # 读取2
        # self.list2 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m2.txt')
        # print('e_by_m2')
        # # 读取3
        # self.list3 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m3.txt')
        # print('e_by_m3')

    # 1 通过别名找到对应的原始实体
    def find_entity(self, dst):
        """
        通过别名找到对应的原始实体
        :param dst:
        :return:
        """
        origin_entitys = []
        find = False
        for l in self.list1:
            for l1 in str(l).split('\t'):
                if l1 == dst:
                    find = True
                    break  # 只可能匹配一个
            if find:
                origin_entitys.append(str(l).split('\t')[0])
                # break # 这里可能会匹配多个

        # print('1/3 %s' % (str(origin_entitys)))

        # origin_entitys_no_repeat = []
        # for e in origin_entitys:
        #     if e not in origin_entitys_no_repeat:
        #         origin_entitys_no_repeat.append(e)
        # assert (len(origin_entitys)!=1)
        ct.print_t("total:%s" % len(origin_entitys))

        return origin_entitys

    # 1 通过别名找到对应的整行实体
    def find_entity_all(self, dst):
        """
        通过别名找到对应的原始实体
        :param dst:
        :return:
        """
        origin_entitys = []
        find = False
        for l in self.list1:
            for l1 in str(l).split('\t'):
                if l1 == dst:
                    find = True
                    break  # 只可能匹配一个

            if find:
                # break
                origin_entitys.extend(str(l).split('\t'))
                # break # 这里可能会匹配多个
            find = False

        # print('1/3 %s' % (str(origin_entitys)))

        # origin_entitys_no_repeat = []
        # for e in origin_entitys:
        #     if e not in origin_entitys_no_repeat:
        #         origin_entitys_no_repeat.append(e)
        # assert (len(origin_entitys)!=1)
        ct.print_t("total:%s" % len(origin_entitys))

        return origin_entitys

    # # 1 通过别名找到对应的原始实体
    # def find_entity(self, dst):
    #     """
    #     通过别名找到对应的原始实体
    #     :param dst:
    #     :return:
    #     """
    #     origin_entitys = []
    #     find = False
    #     for l in self.list1:
    #         for l1 in str(l).split('\t'):
    #             if l1 == dst:
    #                 find = True
    #         if find:
    #             origin_entitys.append(str(l).split('\t')[0])
    #             find = False
    #     print('1/3 %s' % (str(origin_entitys)))
    #
    #     find = False
    #     for l in self.list2:
    #         for l1 in str(l).split('\t'):
    #             if l1 == dst:
    #                 find = True
    #         if find:
    #             origin_entitys.append(str(l).split('\t')[0])
    #             find = False
    #     print('2/3 %s' % (str(origin_entitys)))
    #     find = False
    #     for l in self.list3:
    #         if l == dst:
    #             find = True
    #         if find:
    #             origin_entitys.append(l)
    #             find = False
    #     print('3/3 %s' % (str(origin_entitys)))  # 按长度排序
    #     origin_entitys_no_repeat = []
    #     for e in origin_entitys:
    #         if e not in origin_entitys_no_repeat:
    #             origin_entitys_no_repeat.append(e)
    #     return origin_entitys_no_repeat


    # gzip完整装载需要6分钟,内存不会直接爆炸
    def init_spo(self, f_in="../data/nlpcc2016/demo1/kb.gz"):
        import time
        start_time = time.time()
        if f_in.endswith('.gz'):
            use_gzip = True
        else:
            use_gzip = False
        # exist = os.path.exists(config.par('baike_dict_path'))
        # if exist:
        #     print('加载已经存在的字典')
        #     self.kbqa = ct.pickle_load(config.par('baike_dict_path'))
        #     return
        ct.print_t('init_spo')
        self.kbqa = dict()

        index = 0
        if use_gzip:
            f_out = gzip.open(f_in, 'rb')
        else:
            f_out = codecs.open(f_in, mode="r", encoding="utf-8")
        # with gzip.open(f_in, 'rb') as f_out:
        for line in f_out:
            if use_gzip:
                line = line.decode('utf-8')
            index += 1
            if index % 100000 == 0:
                ct.print_t("%d / 428" % (index / 100000))
                # if (index / 100000)%10 == 0 :
                #     print('collect')
                #     gc.collect()

            ls = str(line).strip('\n').strip('\r').split('\t')
            # s = ct.clean_str_entity(ls[0])
            s = ls[0]
            p = ct.clean_str_rel(ls[1])
            o1 = ls[2]
            t1 = (p, o1)
            # del line
            if s in self.kbqa:
                # try:
                s1 = self.kbqa[s]
                s1.add(t1)
                self.kbqa[s] = s1
            else:
                # except Exception as e1 :
                s1 = set()
                s1.add(t1)
                self.kbqa[s] = s1
        f_out.close()
        # ct.pickle_save(config.par('baike_dict_path'), self.kbqa)
        ct.print_t("init_spo ok")

        time_elapsed = time.time() - start_time
        ct.print_t("time_elapsed: %6.7f" % time_elapsed)

        # ct.pickle_save(config.par('baike_dict_path'), self.kbqa)
        # self.kbqa = d_dict

        # 通过属性值

    #  通过原始实体名，找到对应的所有属性值
    def find_p(self, s, o):
        s1 = self.kbqa.get(s, "")
        if s1 == "":
            return []
        ps = []
        for po in s1:
            if po[1] == o:
                ps.append(po)
        return ps

    #  通过原始实体名，找到对应的所有属性值
    def find_p_by_pos(self, s, o):
        s1 = self.find_e_pos(s)
        if s1 == "":
            return []
        ps = []
        for po in s1:
            if po[1] == o:
                ps.append(po[0])
                break
        return ps

    def record_p_pos(self):
        a1 = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out-1.txt'
        start = 0
        d1 = dict()
        # e_length = 0  # 长度
        line_num = 0
        with open(a1, 'r', encoding='utf-8') as f1:
            for l1 in f1.read().split('$'):
                e1 = l1.split('\t')[0]
                print(e1)
                if e1 in d1:
                    # e_length += len(l1)
                    line_num += 1
                    t1 = d1[e1]
                    d1[e1] = (t1[0], line_num)
                else:
                    # 开始记录下一个
                    line_num = 1
                    # e_length = len(l1)
                    d1[e1] = (start + 1, line_num)
                start += 1
        with codecs.open('../data/nlpcc2016/result/kb_s_pos.txt', 'w', 'utf-8') as o1:
            for k, v in d1.items():
                o1.write("%s\t%s\t%s\n" % (k, v[0], v[1]))
        print(31231321)
        # 试着读取出来

        self.s_pos_dict = d1

    def init_p_pos(self):
        a1 = '../data/nlpcc2016/result/kb_s_pos.txt'
        d1 = dict()
        with open(a1, 'r', encoding='utf-8') as f1:
            for l1 in f1:
                e1 = l1.split('\t')[0]
                start = l1.split('\t')[1]
                line_num = l1.split('\t')[2]
                d1[e1] = (start, line_num)
        self.s_pos_dict = d1
        ct.print_t("init_p_pos")

    def re_writer_kb(self):
        a1 = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out-1.txt'
        a111 = ct.file_read_all_lines_strip(a1)
        with open(a1 + '.1.txt', 'w', encoding='utf-8') as f_o1:
            with codecs.open(a1, 'r', encoding='utf-8') as f_in1:
                for l1 in f_in1.readlines():
                    f_o1.write(l1.replace('\n', "").replace('\r', "") + '$')
        return a1

    def find_e_pos(self, str1):
        a1 = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt'
        v1 = self.s_pos_dict.get(str1, "")
        if v1 == "":
            return ""

        print(v1[0])
        print(v1[1])

        start = 0
        n_l = 0
        l_list = []
        with open(a1, mode='r', encoding='utf-8') as f1:
            for l in f1:
                start += 1
                if start < int(v1[0]):
                    continue
                n_l += 1
                if n_l > int(v1[1]):
                    break
                l_list.append(l)
        # 把list转为spo字典
        # s = ct.clean_str_rn(l_list[0]).split('\t')[0]
        # d1 = dict()
        s1 = set()
        for l in l_list:
            p = ct.clean_str_rn(l).split('\t')[1]
            o = ct.clean_str_rn(l).split('\t')[2]
            s1.add((p, o))
        # d1[s] = s1
        return s1

    # 重新输出实体-长度，并排序,
    @staticmethod
    def statistics_subject_extract(f_in="../data/nlpcc2016/extract_entitys.txt",
                                   f_out="../data/nlpcc2016/extract_entitys_statistics.txt"):

        d_dict = dict()
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    ls = line.split('\t')
                    for l1 in ls:
                        if l1 in d_dict:
                            d_dict[l1] += 1
                        else:
                            d_dict[l1] = 1
            tp = ct.sort_dict(d_dict, True)
            for t in tp:
                out.write("%s\t%s\n" % (t[0], t[1]))

        print(5435436)

    @staticmethod
    def build_vocab_cc():
        counter = ct.generate_counter()
        word_set = set()
        fs = ['../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt',
              '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt']
        fs1 = ['../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt.alais_relations_1.txt.out_e_r_combine-1.txt',
               '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt.alais_relations_1.txt.out_e_r_combine-1.txt']
        fs2 = []
        for f_in in fs2:
            print(f_in)
            # continue

            with open(f_in, mode='r', encoding='utf-8') as rf:
                for l in rf.readlines():
                    now = counter()
                    if now % 10000 == 0:
                        ct.print_t(now / 10000)
                    for w in l:
                        word_set.add(w)
        # tc = TextConverter(word_set, max_vocab=999999999999)
        tc = TextConverter(word_set, filename='../data/nlpcc2016/demo1/nlpcc2016.vocab')
        print(tc.vocab_size)
        # for w in tc.vocab:
        #     ct.print_t(w)
        tc.save_to_file_raw('../data/nlpcc2016/demo1/nlpcc2016.vocab_raw.txt')

        # @staticmethod
        # def convert_vocab_cc():
        #     tc = TextConverter(filename='../data/nlpcc2016/demo1/nlpcc2016.vocab')
        #     for w in tc.vocab:
        #         try:
        #         except Exception as e1:
        #             print(e1)

    @staticmethod
    def load_vocab_cc():
        tc = TextConverter(filename='../data/nlpcc2016/demo1/nlpcc2016.vocab')
        print(tc.vocab_size)

    @staticmethod
    def prodeuce_embedding_vec_file(filename):
        from gensim import models
        f1 = '../data/nlpcc2016/demo1/'
        converter = TextConverter(filename='../data/nlpcc2016/demo1/nlpcc2016.vocab')
        model = models.Word2Vec.load(filename)
        # 遍历每个单词，查出word2vec然后输出
        v_base = model['结']
        ct.print(v_base)
        for word in converter.vocab:
            try:
                # if word == ' ':
                #     word = '结'
                # w1 = word
                # word = Converter('zh-hans').convert(word)
                # if word != w1:
                #     # print(w1)
                #     ct.just_log(f1 + "wiki.vector3", w1)
                v = model[word]
            except Exception as e1:
                msg1 = "%s : %s " % (word, e1)
                ct.print(msg1)
                ct.just_log(f1 + "wiki.vector2.log", msg1)
                v = model['结']
            m_v = ' '.join([str(x) for x in list(v)])
            msg = "%s %s" % (word, str(m_v))
            # ct.print(msg)
            ct.just_log(f1 + "wiki.vector2", msg)
            # msg = "%s %s" % ('end', str(v_base))
            # ct.just_log(f1 + "wiki.vector2", msg)

    # 读取实体所有的实体    返回所有的关系集合
    def read_entity_and_get_all_neg_relations_cc(self, entity_id, ps_to_except):
        e_s = self.kbqa.get(str(entity_id).lower(), "")
        if e_s == "":
            print(entity_id)
            raise Exception('entity cant find')
        r1 = []
        for s1 in e_s:
            if s1[0] not in ps_to_except:
                r1.append(s1[0])
        return r1

    # 输入识别结果，输出匹配R2格式
    # 《机械设计基础》这本书的作者是谁？    杨可桢，程光蕴，李仲生
    # 机械设计基础         作者          杨可桢，程光蕴，李仲生
    # 问题0 答案1 实体s-2 关系p-3 属性值o-4
    @staticmethod
    def rebulild_qa_rdf():

        f1 = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
        f2 = '../data/nlpcc2016/demo1/r3.txt'
        f3 = '../data/nlpcc2016/demo1/qa_rdf.txt'

        l1s = ct.file_read_all_lines_strip(f1)
        l2s = ct.file_read_all_lines_strip(f2)

        with open(f3, mode='w', encoding='utf-8') as f33:
            for index in range(len(l2s)):
                l1 = l1s[index]
                l2 = l2s[index]
                f33.write(l1 + '\t' + str(l2).replace('NULL', '$') + "\n")

    # 排序算法
    def get_total(self, word):
        v1 = dict(self.d1).get(word, 0)
        v1 = v1 * 100 - len(word)  # 相同个数看文字长度
        return v1

    def get_qiwang(self, word):
        v1 = dict(self.d_f6s).get(word, 0)
        return v1

    # 看看m1_2 中是否有重复的
    def test_m1_m2(self):
        m1s = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_12.txt')
        xs = []
        for x in m1s:
            xs.extend(str(x).split('\t'))

        print(len(xs))
        print(len(set(xs)))

    # 合并m1和m2文件
    # v2 删除了空格，转换小写,输出到n_gram
    # V3 不过滤没有别名的，没有别名的设置其别名为自己
    def combine_m1_m2(self, f1='../data/nlpcc2016/result/e_by_m1.txt',
                      f2='../data/nlpcc2016/result/e_by_m2.txt',
                      f3='../data/nlpcc2016/n_gram/e_12.txt'):
        # 读取出来
        m1s = ct.file_read_all_lines_strip(f1)
        m2s = ct.file_read_all_lines_strip(f2)
        m1s = [str(x).lower().replace(' ', '') for x in m1s]
        m2s = [str(x).lower().replace(' ', '') for x in m2s]
        #
        # 将2中的每个实体逐个遍历1，如果存在则添加进1
        # m1 白鳞环锈伞	白鳞环伞	pholiota destruens (brond) gill.	杨环锈伞
        # m2 mt(肉盾)	肉盾
        d1 = dict()
        # 装载
        for m1_line in m1s:
            m1_es = str(m1_line).split('\t')
            if len(m1_es) == 1:  # 过滤没有别名的
                # print(m1_line)
                # continue
                print(m1_es[0])
                d1[m1_es[0]] = set([m1_es[0]])
            else:
                # m1_es_1=m1_es[1:]
                #    for x in m1_es:
                d1[m1_es[0]] = set([x for x in m1_es[1:]])
        # 检查
        for m2_line in m2s:
            m2_es = str(m2_line).split('\t')
            exist = False
            key = ''
            for m2_e in m2_es:
                if m2_e in d1:
                    exist = True
                    key = m2_e
                    break
            if exist:  # 如果m2e 存在于m1中 则整个添加到 1 中
                s1 = d1[key]
                for m2_e in m2_es:
                    s1.add(m2_e)
                d1[key] = s1
        # 输出整个字典
        # d2 = ct.sort_dict(d1)
        with open(f3, mode='w', encoding='utf-8') as o1:
            for k, v in d1.items():
                msg = "%s\t%s\n" % (k, '\t'.join(v))
                o1.write(msg)
                # ct.just_log('../data/nlpcc2016/result/e_12.txt',msg)

    # F0.1.2 压缩成gz
    @staticmethod
    def gzip_file(f2='../data/nlpcc2016/demo1/kb.gz',
                  f1='../data/nlpcc2016/demo1/kb.txt'):
        cgc = ct.generate_counter()
        with gzip.open(f2, 'wb') as f_out:
            with open(f1, 'r', encoding='utf-8') as f_in:
                for l in f_in:
                    i = cgc()
                    if i % 10000 == 0:
                        print(i / 10000)
                    f_out.write(l.encode('utf-8'))
        print('ok')


class baike_test:
    @staticmethod
    def try_idf(f1='../data/nlpcc2016/ner_t1/extract_entitys_all.txt',
                f2='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.statistics.txt',
                f3='../data/nlpcc2016/ner_t1/q.rdf.txt'):
        # idf = log(n / docs(W, D))
        # 即文档总数n与词w所出现文件数docs(W, D)比值的对数
        f1s = ct.file_read_all_lines_strip(f1)
        # 读取是正确的实体
        f3s = ct.file_read_all_lines_strip(f3)

        d1 = dict()
        # 统计
        for words_list in f1s:
            for word in str(words_list).split('\t'):
                if word in d1:
                    d1[word] += 1
                else:
                    d1[word] = 1
        print(321)
        d3 = dict()
        # 统计
        for words_list in f3s:
            if len(str(words_list).split('\t')) < 3:
                continue
            word = str(words_list).split('\t')[2]
            if word in d3:
                d3[word] += 1
            else:
                d3[word] = 1

        # 排序
        total = len(f1s)
        print(total)
        list1 = ct.sort_dict(d1)
        with open(f2, mode='w', encoding='utf-8') as out:
            out.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ('实体', '出现次数', '出现次数/总数', 'IDF', '命中次数', '期望'))
            for t in list1:
                out.write("%s\t%s\t%f\t%f\t%s\t%f\n" % (
                    t[0], t[1], t[1] / total, math.log(t[1] / total), d3.get(t[0], 0),
                    (d3.get(t[0], 0) * 10000 / t[1])))
        print(11111132)

    # 使用jieba分词，对每一个候选词做词性标注，1 0标识
    @staticmethod
    def try_jieba(f1='../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                  f2='../data/nlpcc2016/ner_t1/extract_entitys_all.txt',
                  f3='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.identify.txt',
                  f4='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.jieba.txt'
                  ):

        import jieba.posseg as pseg
        # import jieba
        # jieba.set_dictionary('../data/jieba_dict/dict.txt.big')
        #
        # # load stopwords set
        # stopwordset = set()
        # with open('../data/jieba_dict/stopwords.txt', 'r', encoding='utf-8') as sw:
        #     for line in sw:
        #         stopwordset.add(line.strip('\n'))

        word_set = set()
        f1s = ct.file_read_all_lines_strip(f1)
        f1s = [str(x).split('\t')[0] for x in f1s]
        f2s = ct.file_read_all_lines_strip(f2)

        # 做好词性标注后使用三个下划线分割 ___
        f3s = []
        f4s = []
        i = -1
        for sentence in f1s:
            i += 1
            # print(w.word,w.flag)
            pseg_words = pseg.cut(sentence=sentence)
            pseg_words_list = []
            for t1 in pseg_words:
                pseg_words_list.append((t1.word, t1.flag))
            pseg_words = pseg_words_list

            f3s.append(pseg_words)  # 等会输出
            f2s_line = str(f2s[i]).split('\t')
            f2s_line_new = []
            for f2s_line_word in f2s_line:
                find = False
                find_pseg_w = ''
                for pseg_w in pseg_words:
                    if pseg_w[0] == f2s_line_word:
                        find = True
                        find_pseg_w = pseg_w
                        break
                if find:
                    # t1 = (find_pseg_w.word, find_pseg_w.flag)
                    f2s_line_new.append(find_pseg_w)
                else:
                    t1 = (f2s_line_word, 'NULL')
                    f2s_line_new.append(t1)
            f4s.append(f2s_line_new)
        # 输出词性标注文件
        i = -1
        with open(f3, mode='w', encoding='utf-8') as o1:
            for f2s_line_new in f3s:
                i += 1
                o1.write(f1s[i] + '\t')
                msg = ''
                for t1 in f2s_line_new:
                    msg += "%s___%s\t" % (t1[0], t1[1])
                o1.write(msg + '\n')
        print(33213)
        i = -1
        with open(f4, mode='w', encoding='utf-8') as o1:
            for f2s_line_new in f4s:
                i += 1
                o1.write(f1s[i] + '\t')
                msg = ''
                for t1 in f2s_line_new:
                    msg += "%s___%s\t" % (t1[0], t1[1])
                o1.write(msg + '\n')



                # 标注已经分词的文本并做输出

    # 合并识别结果和未识别,一次性
    @staticmethod
    def one_combine_all(f1='../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                        f2='../data/nlpcc2016/ner_t1/q.rdf.txt',
                        f3='../data/nlpcc2016/ner_t1/r2.txt'):
        # 《高等数学》是哪个出版社出版的？	武汉大学出版社

        q_s = ct.file_read_all_lines_strip(f1)
        r2_s = ct.file_read_all_lines_strip(f3)
        # r2_s_no = ct.file_read_all_lines_strip('../data/nlpcc2016/ner_t1/r2_no.txt')
        new_ls = []
        index1 = 0
        index2 = 0
        for i in range(len(q_s)):
            if str(q_s[i]).split('\t')[0] == str(r2_s[index1]).split('\t')[0]:

                new_ls.append(r2_s[index1])
                index1 += 1
            else:
                new_ls.append(q_s[i] + '\t' + 'NULL')
        with open(f2, mode='w', encoding='utf-8') as o1:
            for item in new_ls:
                o1.write(str(item).replace(' -> ', '\t').replace('\t\t', '\t') + '\n')

    # 试试检测一下前1，前2，前3命中的概率
    # 增加别名词典匹配
    # @staticmethod
    def try_test_acc_of_m1(self, f1='../data/nlpcc2016/ner_t1/q.rdf.txt',
                           f2='../data/nlpcc2016/ner_t1/q.rdf.txt.failed4.txt',
                           f3='../data/nlpcc2016/ner_t1/extract_entitys_all.txt',
                           f4='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.jieba.txt',
                           f5='../data/nlpcc2016/ner_t1/q.rdf.filter.txt',
                           f6='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.statistics.txt',
                           reweiter=False,
                           use_cx=False,
                           use_expect=False,
                           acc_index=[1, 2, 3]):
        # f1   # 输入文件
        # 《机械设计基础》	机械设计基础	设计基础	机械设计	机械	基础	这本书	作者	本书	设计	谁？	是谁
        # f2 = f1 + '.failed.txt'  # 输出文件
        # f3= '../data/nlpcc2016/ner_t1/extract_entitys2.txt'  # 抽取的结果
        bkh = baike_helper()
        bkh.init_find_entity()
        # 《高等数学》是哪个出版社出版的？	武汉大学出版社	高等数学(微积分)	 出版社	 武汉大学出版社

        #
        acc1 = 0.0
        f3s = ct.file_read_all_lines_strip(f3)
        f1s = ct.file_read_all_lines_strip(f1)
        f4s = ct.file_read_all_lines_strip(f4)  # 结巴 标注好的分词
        f6s = ct.file_read_all_lines_strip(f6)  # 期望.IDF文件
        f5s = []
        # 载入期望 5.9
        f6s = f6s[1:]
        d_f6s = dict()
        for f6s_line in f6s:
            key = str(f6s_line).split('\t')[0]
            v = float((f6s_line).split('\t')[5])
            d_f6s[key] = v
        bkh.d_f6s = d_f6s
        ##
        ct.print_t('start')
        # 取第一个与正确答案做比较成功+1错误不加
        # acc_index = [ 3]
        total = len(f1s)
        total2 = 0
        acc = dict()
        record = []
        for l_i in acc_index:
            acc[str(l_i)] = 0

        skip = 0
        cgc = ct.generate_counter()
        for i in range(len(f1s)):
            index = cgc()
            if index % 10 == 0:
                print(index / 10)
            if len(str(f1s[i]).split('\t')) < 3:
                skip += 1
                # print(f1s[i])
                continue
            if str(f1s[i]).__contains__('NULL'):
                skip += 1
                # print(f1s[i])
                continue

            total2 += 1

            f1s_i_e = str(f1s[i]).split('\t')[2]
            f1s_i_e = ct.clean_str_zh2en(f1s_i_e)  # 符号转换
            # ct.print_t('答案改写前:%s' % f1s_i_e)
            f1s_i_e = f1s_i_e.lower().replace(' ', '')
            f1s_i_e = baike_helper.entity_re_extract_one_repeat(f1s_i_e)  # 小写
            # if (f1s_i_e.__contains__('《') and f1s_i_e.__contains__('》')) or \
            #         (f1s_i_e.__contains__('(') and f1s_i_e.__contains__(')')):
            #     f1s_i_e = baike_helper.entity_re_extract_one(f1s_i_e).lower()  # 小写
            # if (f1s_i_e.__contains__('《') and f1s_i_e.__contains__('》')) or \
            #         (f1s_i_e.__contains__('(') and f1s_i_e.__contains__(')')):
            #     f1s_i_e = baike_helper.entity_re_extract_one(f1s_i_e).lower()  # 小写
            # f1s_i_e2 = ct.clean_str_zh2en(f1s_i_e2) # 符号转换
            f1s_i_e2 = f1s_i_e
            # ct.print_t('答案改写后:%s' % f1s_i_e)
            # if f1s_i_e == 'tcl':
            #     print(555555)

            filter_flags = ['ul', 'tg', 'an', 'vq', 'e', 'c', 'ag', 'u', 'mq', 'df', 'vd', 'ug', 'f']

            for l_i in acc_index:
                start_list = str(f3s[i]).split('\t')
                if use_cx:
                    # 增加词性
                    f4s_line = str(f4s[i]).split('\t')
                    d_f4s_line = dict()
                    for f3s_line_word in f4s_line:
                        # print(f3s_line_word)
                        if len(f3s_line_word.split('___')) < 2:
                            continue
                        word_1 = f3s_line_word.split('___')[0]
                        word_proterty = f3s_line_word.split('___')[1]
                        # s f v a x u
                        if word_proterty in filter_flags:
                            d_f4s_line[word_1] = False
                        else:
                            d_f4s_line[word_1] = True
                            #

                # ct.print_t('候选去除前:%s' % start_list)
                list1 = [x.lower().replace(' ', '') for x in start_list]  # 小写
                # 去掉候选的书名号和括号
                list1 = [baike_helper.entity_re_extract_one_repeat(ct.clean_str_zh2en(x)) for x in list1]
                # 去重
                # list1_new = []
                # for _ in list1:
                #     if _ not in list1_new and _ != '':
                #         # 增加词性过滤
                #         need_add = d_f4s_line.get(_, True)
                #         if need_add:
                #             list1_new.append(_)
                if use_cx:
                    list1_new = ct.list_no_repeat_cx(list1, d_f4s_line)
                else:
                    list1_new = ct.list_no_repeat(list1)
                # 5.8.3 去掉词语包含试试 有一首歌叫	有一首歌	一首歌
                # 不好使
                # list1_new_2 = []
                # for list1_new_word in list1_new:
                #     if not ct.be_contains(list1_new_word,list1_new):
                #         list1_new_2.append(list1_new_word)
                # list1_new = list1_new_2

                start_list = list1_new
                if use_expect:
                    min1 = min(len(start_list), 15)  # 最多排序6个
                    start_list = start_list[0:min1]
                    # 通过期望 重写排序 候选 5.9
                    start_list = sorted(start_list, key=bkh.get_qiwang, reverse=True)

                # ct.print_t('候选去除后:%s' % start_list)
                list1 = start_list[0:l_i]
                # list1 = [ct.clean_str_zh2en(x.lower()) for x in list1] # 小写


                # 重写这里 将list1中的别名词一起扩进来
                # ct.print_t('扩展前:%s'%list1)
                # ori_list = []
                # for list1_1 in list1:
                #     ori_list.extend( bkh.find_entity_all(list1_1))# 找出来
                # list1.extend( ori_list)
                # list1 = list(set(list1))
                # #
                # ct.print_t('扩展后:%s' % list1)

                exist = f1s_i_e in list1 or f1s_i_e2 in list1
                if exist:
                    acc[str(l_i)] += 1
                elif l_i == 3 and not exist:
                    if str(f1s[i]).split('\t')[0] in ['有一本叫《毛泽东》的书是怎样装订的'
                        , '《兄弟》属于哪种小说', '《i》是什么音乐风格的？',
                                                      '《因为我爱你》是怎样装帧的'
                                                      ]:
                        print(1200000)
                elif l_i == 999:
                    if str(f1s[i]).split('\t')[0] in [
                        '请问荣耀xl是什么时候曝光的？'
                    ]:
                        print(333333333333)

                    record.append("%s\t%s" % (f1s[i], f3s[i]))
                    # list1 = str(f3s[i]).split('\t')[0:3]
                    # exist = f1s_i_e in list1 or f1s_i_e2 in list1
                    #     # 记录下来 分析一下
                    # if not exist:
                    #     record.append("%s\t%s" % (f1s[i], f3s[i]))

        print("skip:%d total:%d  toatal2:%d " % (skip, total, total2))

        for k, v in acc.items():
            print("前%s,get:%d   acc: %f,total - skip=%d  " % (k, v, v / (total - skip), total - skip))
        print(len(record))
        with open(f2, mode='w', encoding='utf-8') as o1:
            for item in record:
                o1.write(item + '\n')

    # 一次性 合并
    @staticmethod
    def file_combine(f1='../data/nlpcc2016/ner_t1/extract_entitys2.txt',
                     f2='../data/nlpcc2016/ner_t1/extract_entitys2_1gram.txt',
                     f3='../data/nlpcc2016/ner_t1/extract_entitys_all.txt'):
        l1s = ct.file_read_all_lines_strip(f1)
        l2s = ct.file_read_all_lines_strip(f2)

        with open(f3, mode='w', encoding='utf-8') as o1:
            for i in range(len(l1s)):
                msg = "%s\t%s\n" % (l1s[i], l2s[i])
                o1.write(msg)

    # 1
    # 一次性 替换带空格的分词进不带的
    @staticmethod
    def file_combine_space(f1='../data/nlpcc2016/ner_t1/extract_entitys_v3.txt', # 原始
                     f2='../data/nlpcc2016/ner_t1/extract_entitys_v3-1.txt',  # 新
                     f3='../data/nlpcc2016/ner_t1/extract_entitys_all.txt'):
        l1s = ct.file_read_all_lines_strip(f1)
        l2s = ct.file_read_all_lines_strip(f2)

        with open(f3, mode='w', encoding='utf-8') as o1:
            for i in range(len(l1s)):
                if str(l2s[i]).__contains__('#THE_SAME#'):
                    msg = "%s\n" % (l1s[i])
                else:
                    msg = "%s\n" % (l2s[i])
                o1.write(msg)

    # file_tj
    @staticmethod
    def file_tj(f1='../data/nlpcc2016/ner_t1/extract_entitys_all.txt', # 原始
                f_out='../data/nlpcc2016/ner_t1/extract_entitys_all_tj.txt' ):
        result = ct.file_read_all_lines_strip(f1)
        # l2s = ct.file_read_all_lines_strip(f2)

        # 将统计出现的次数，按出现次数少的排在前面
        d1 = dict()
        for words_list in result:
            for word in words_list:
                if word in d1:
                    d1[word] += 1
                else:
                    d1[word] = 1
        # result = [x= sorted(x,key=get_total(x))  for x  in  result]
        bkh.d1 = d1
        for index in range(len(result)):
            tmp = result[index]
            tmp = sorted(tmp, key=bkh.get_total)
            result[index] = tmp

        with open(f_out, mode='w', encoding='utf-8') as o1:
            for words_list in result:
                # print("------")
                # print(x)
                o1.write("%s\n" % '\t'.join(words_list))
                # for x1 in x :
                #     print("%s  %s"%(x1,bkh.get_total(x1)))

    # 统计前3的词性
    @staticmethod
    def try_sta_jieba(f1='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.jieba.txt',
                      f2='../data/nlpcc2016/ner_t1/extract_entitys_all.txt.jieba.txt.tj_3.txt',
                      f3='../data/nlpcc2016/ner_t1/q.rdf.txt',
                      only_right=False):
        f1s = ct.file_read_all_lines_strip(f1)
        f3s = ct.file_read_all_lines_strip(f3)
        d1 = dict()
        i = -1
        for f1s_line in f1s:
            i += 1
            f1s_line = str(f1s_line).split('\t')[1:5]

            if len(str(f3s[i]).split('\t')) < 3:
                print(f3s[i])
                continue

            right1 = str(f3s[i]).split('\t')[2]  # 先不做处理
            right2 = baike_helper.entity_re_extract_one_repeat(right1.lower())  # 小写

            ct.print_t("diff:%s \t %s" % (right1, right2))
            if right1 == '王娟(全国劳动模范)':
                print(32131)
            for w in f1s_line:
                cx = w.split('___')[1]
                if only_right:
                    # if right1.lower() != w.split('___')[0].lower():
                    # 不在2个候选答案里面
                    cand_e1 = w.split('___')[0].lower()
                    cand_e2 = \
                        baike_helper.entity_re_extract_one_repeat(ct.clean_str_zh2en(cand_e1))
                    ct.print_t("diff2:%s \t %s" % (cand_e1, cand_e2))
                    if cand_e1 == '王娟':
                        print(543543)
                    if cand_e2 not in [right1, right2]:
                        continue

                if cx in d1:
                    d1[cx] += 1
                else:
                    d1[cx] = 1

        list1 = ct.sort_dict(d1)
        with open(f2, mode='w', encoding='utf-8') as o1:
            for l in list1:
                # msg = "%s\t%s" % (l[0], l[1])
                msg = "%s\t" % (l[0])
                o1.write(msg + '\n')
        return d1


# F2.3 空格分割
def seg_m():
    bk = baike_helper()
    f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt"
    f_out = f_in + "-out.txt"
    bk.convert_text_to_seg(f_in, f_out, type="questions")


def n_gram_math_all(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt",
                    f_out='../data/nlpcc2016/result/extract_entitys2.txt',
                    f3="../data/nlpcc2016/result/combine_e12.txt.statistics.txt"):
    bkh = baike_helper()
    bkh.init_ner(f_in2=f3)

    index = 0
    result = []
    with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
        for line in read_file:
            index += 1
            # if index > 10:
            #     break
            print(index)
            s2 = line.replace("\r", "").replace("\n", "").split("\t")[0]
            s = line.replace("\r", "").replace("\n", "").replace(' ', '').split("\t")[0]
            if s == s2:
                ss = ['#THE_SAME#']
                result.append(ss)
                continue

            ss = bkh.ner(s)
            if len(ss) > 0:
                # ct.just_log("../data/nlpcc2016/extract_entitys2.txt", '\t'.join(ss))
                result.append(ss)
            else:
                ss = ['NULL']
                result.append(ss)
                # ct.just_log("../data/nlpcc2016/extract_entitys2.txt", "NULL")
            print(ss)
    # 将统计出现的次数，按出现次数少的排在前面
    # d1 = dict()
    # for words_list in result:
    #     for word in words_list:
    #         if word in d1:
    #             d1[word] += 1
    #         else:
    #             d1[word] = 1
    # # result = [x= sorted(x,key=get_total(x))  for x  in  result]
    # bkh.d1 = d1
    # for index in range(len(result)):
    #     tmp = result[index]
    #     tmp = sorted(tmp, key=bkh.get_total)
    #     result[index] = tmp

    with open(f_out, mode='w', encoding='utf-8') as o1:
        for words_list in result:
            # print("------")
            # print(x)
            o1.write("%s\n" % '\t'.join(words_list))
            # for x1 in x :
            #     print("%s  %s"%(x1,bkh.get_total(x1)))


# def find_r_all():
#     bkh = baike_helper()
#     # bkh.init_ner()
#     bkh.init_spo()
#     f_in = "../data/nlpcc2016/extract_entitys.txt"
#     index = 0
#     # 还差一个所有的o
#     with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
#         for line in read_file:
#             index += 1
#             print(index)
#             e_list = line.replace("\r", "").replace("\n", "").split("\t")
#
#             # for e in e_list:
#
#             # ss = bkh.ner(s)
#             # if len(ss) > 0:
#             #     ct.just_log("../data/nlpcc2016/extract_entitys.txt", '\t'.join(ss))
#             # else:
#             #     ct.just_log("../data/nlpcc2016/extract_entitys.txt", "NULL")
#             print(1)


def find_all_ps_2_6_3():
    bh = baike_helper()
    ct.print_t(1)
    bh.init_find_entity()
    ct.print_t(2)
    is_debug = False
    if is_debug:
        bh.init_spo('../data/nlpcc2016/demo2/kb.txt')
        f_q_in = '../data/nlpcc2016/demo2/r2.txt'
        f_cand_q_in = '../data/nlpcc2016/demo2/extract_entitys.txt'
    else:
        bh.init_spo(f_in=config.par('cc_kb_path_full'))
        f_q_in = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
        f_cand_q_in = '../data/nlpcc2016/extract_entitys.txt'

    # bh.init_ner()

    ct.print_t(3)

    cand_s = ct.file_read_all_lines_strip(f_cand_q_in)
    ct.print_t(4)
    index = -1
    filter_words = ['请问', '什么时候', '什么', '是谁', '有谁知道', '谁知道', '谁能告诉我', '还有什么', '什么？', ]
    with open(f_q_in, mode='r', encoding='utf-8') as rf:
        for l in rf.readlines():
            index += 1
            if index < config.get_find_index():
                continue
            ct.print_t(index)
            ct.print_t(l)
            o = ct.clean_str_rn(l.split('\t')[1])  # 实体
            ss = str(cand_s[index]).split('\t')  # N-GRAM 实体名
            s_set = set()  # 候选实体集合
            ps = []  # 可能的 属性集合 list<tuple(p,o)>
            es = []  # 实体集合 list<entity>
            ps_find = ''
            es_find = ''

            find_r = False
            for s1 in ss:
                if len(s1) == 1 or s1 in filter_words:
                    print(s1)
                    continue
                ps1 = bh.find_p(s1, o)  # ps1 = tuple(p,o)
                if len(ps1) > 0:
                    ct.print_t("%s\t%s" % (s1, o))
                    es.append(s1)
                    ps.extend(ps1)

                    # 查找就停止
                    find_r = True
                    break

            if find_r == False:
                for s1 in ss:
                    s1_result = bh.find_entity(s1)
                    for s11 in s1_result:  # 之前查过的不再查了
                        if s11 in ss:
                            continue
                        # s_set.add(s11)
                        # # 找出所有可能的实体
                        # for s2 in s_set:
                        # if s11 == '机械设计基础(2010年高等教育出版社出版作者杨可桢)':
                        #     print(532424)
                        # 不加载
                        # ps1 = bh.find_p_by_pos(s11, o)
                        # 全加载
                        ps1 = bh.find_p(s11, o)
                        if len(ps1) > 0:
                            ct.print_t("%s\t%s" % (s11, o))
                            es.append(s11)
                            ps.extend(ps1)
                            # 查找就停止
                            find_r = True
                            break
                    if find_r:
                        break

            # 输出所有可能的关系
            if len(ps) == 0:
                ps = ['NULL']
                ct.just_log('../data/nlpcc2016/result/ps.txt', "%s\t%s\t%s\t%d" % ('NULL', 'NULL', o, index))
            else:
                ct.just_log('../data/nlpcc2016/result/ps.txt',
                            "%s\t%s\t%s\t%d\t%d" % (es[0], ps[0][0], o, index, len(ps)))
            ct.print_t(ps)


def extract_not_use_cx():
    global a
    d1 = baike_test.try_sta_jieba(only_right=True)
    d2 = baike_test.try_sta_jieba(only_right=False)
    aa = []
    for k1, v1 in d2.items():
        if k1 not in d1:
            aa.append(k1)
    msg = ''
    for a in aa:
        msg += '\'%s\',' % a
    print(msg)


if __name__ == '__main__':

    bkt = baike_test()
    bkh = baike_helper()
    # baike-test流程 3.0
    # baike_test.one_combine_all(f1='../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
    #                            f2='../data/nlpcc2016/tang/20180205-1/q.rdf.txt',
    #                            f3='../data/nlpcc2016/tang/20180205-1/shibiejieguo1.txt')
    # F0.1.2
    # baike_helper.gzip_file()

    # F0.1.3
    # bkh.extract_kb(f1='../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt',
    #                f2="../data/nlpcc2016/demo1/kb?.txt",
    #                f3='../data/nlpcc2016/demo1/q.rdf.txt')

    # 3.1

    # a = '《兄弟》(小说)'
    # a = '《因为我爱你》(推理小说)'
    # a = 'HTC One(2012年HTC手机系列(HTC One X,S,Ⅴ))'
    # a = '2011年《金融时报》全球500强排名 (101-200)'
    # a = '卓越号航空母舰(1982年无敌级航空母舰次舰卓越号(R06))'
    # a = '燕国(周朝诸侯国燕(yān)国)'

    # b = baike_helper.entity_re_extract_one_repeat(a)
    # print(b)

    # c = baike_helper.entity_re_extract_one(b)
    # print(c)

    if True:
        bkt.try_test_acc_of_m1(
            f3='../data/nlpcc2016/ner_t1/extract_entitys_v3.txt',
            # extract_entitys_v3                extract_entitys_all
            f2='../data/nlpcc2016/ner_t1/q.rdf.txt.failed_v3-1.txt',
            use_cx=False, use_expect=False, acc_index=[999])

    # baike_helper.e_r_combine()
    # method_name()
    # baike_helper.statistics_subject_len()
    # build_and_statistics_vocab()
    # s ="《机械设计基础》这本书的作者是谁"
    # method_name2()

    # bkh = baike_helper()
    # 2.5.1 r_combine
    # baike_helper.r_combine()

    # 2.5.2
    # 重新抽取m2
    # baike_helper.entity_re_extract()
    # fs = ['../data/nlpcc2016/result/e_by_m1.txt', '../data/nlpcc2016/result/e_by_m2.txt']
    # out = '../data/nlpcc2016/result/combine_e12.txt'
    # baike_helper.combine_all_entitys(fs, out)
    # f_in = '../data/nlpcc2016/result/combine_e12.txt'
    # f_out = f_in + ".statistics.txt"
    # baike_helper.statistics_subject_len(f_in, f_out)

    # 2.5.3
    # 合并m1和m2文件
    # bkh.combine_m1_m2(f1='../data/nlpcc2016/result/e_by_m1.txt',
    #                   f2='../data/nlpcc2016/result/e_by_m2.txt',
    #                   f3='../data/nlpcc2016/n_gram/e_12.txt')

    # 2.5.4.1 按长度重写
    # f_in = '../data/nlpcc2016/n_gram/e_12.txt'
    # f_out = f_in + ".tj.txt"
    # baike_helper.statistics_subject_len(f_in, f_out)

    # 2.5.4.2 按长度排序
    # f_in = '../data/nlpcc2016/n_gram/e_12.txt'
    # f_out = f_in + ".tj_sort.txt"
    # baike_helper.statistics_and_sort_subject_by_len(f_in, f_out)

    # 2.5.5 n-gram匹配
    # n_gram_math_all(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt",
    #                 f_out='../data/nlpcc2016/result/extract_entitys2.txt',
    #                 f3="../data/nlpcc2016/result/combine_e12.txt.statistics.txt")

    # 测试
    # ct.print_t("start")
    # bkh = baike_helper()
    # # bkh.combine_m1_m2()
    # bkh.test_m1_m2()

    # bkh.init_ner()
    # bkh.init_spo(f_in='../data/nlpcc2016/demo1/kb.gz')
    # print(bkh.kbqa['高等数学'])


    # 2.6.1 通过实体找到原始实体
    # a = baike_helper()
    # a.init_find_entity()
    # b = a.find_entity('红楼梦')
    # print(b)

    # 2.6.2
    # a = baike_helper()
    # ct.print_t("init_p_pos")
    # a.init_p_pos()
    # print("init_p_pos")
    # ct.print_t("begin")
    # b = a.find_p_by_pos('机械设计基础(2010年高等教育出版社出版作者杨可桢)', '杨可桢，程光蕴，李仲生')
    # print(b)
    # ct.print_t("end")


    # 2.6.3 通过关系确定o
    # 读取问题、候选实体，通过2.6.1找到原始实体，通过2.6.2找到对应的关系，输出所以可能的关系

    # find_all_ps_2_6_3()

    # 2.7 统计
    # baike_helper.statistics_subject_extract()




    # a = baike_helper()
    # a.record_p_pos()

    # 3.1
    # baike_helper.build_vocab_cc()
    # word2vecbin_file = '../data/nlpcc2016/demo1/wiki_texts_seg_by_space.txt.bin'
    # baike_helper.prodeuce_embedding_vec_file(word2vecbin_file)

    # baike_helper.rebulild_qa_rdf()

    # baike_helper.load_vocab_cc()


    # 合并 5.6.1
    # bkt.file_combine()
    # 重写生成一些N-GRAM 5.6.2
    # 5.6.3 合并space的
    bkt.file_combine_space()
    # 5.6.4 统计完重写输出
    bkt.file_tj()

    # N元分词全部
    if False:
        n_gram_math_all(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt",
                        f_out='../data/nlpcc2016/n_gram/extract_entitys_v3.txt',
                        f3="../data/nlpcc2016/n_gram/e_12.txt.tj_sort.txt")
    # 测试不能匹配的
    if False:
        n_gram_math_all(f_in="../data/nlpcc2016/ner_t1/q.rdf.txt.failed-n-gram.txt",
                        f_out='../data/nlpcc2016/ner_t1/extract_entitys-n-gram.txt',
                        f3="../data/nlpcc2016/n_gram/e_12.txt.tj_sort.txt")

    # 5.9
    # baike_test.try_idf()
    # 5.8
    # baike_test.try_jieba()
    # 5.8.2
    # extract_not_use_cx()

    ct.print_t("finsih ")
