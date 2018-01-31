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
from gensim import models
from lib.converter.langconv import *
from lib.config import config


class baike_helper:
    # def __init__(self):
    #     # jieba.set_dictionary('../data/jieba_dict/dict.txt.big')
    #     # self.stopwordset = set()
    #     # with open('../data/jieba_dict/stopwords.txt', 'r', encoding='utf-8') as sw:
    #     #     for line in sw:
    #     #         self.stopwordset.add(line.strip('\n'))
    #     print(1)

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
    # def convert_text_to_seg(self, file_in, file_out, type="rdf"):
    #     with open(file_out, 'w', encoding='utf-8') as f_out:
    #         with open(file_in, 'r', encoding='utf-8') as f_in:
    #             for line in f_in:
    #                 if type == "rdf":
    #                     # 增加操作
    #                     print(124444)
    #                 if type == "questions":
    #                     line = line.split('\t')[0]
    #                 line = line.strip('\n')
    #                 words = jieba.cut(line, cut_all=False)
    #
    #                 words_out = []
    #                 for word in words:
    #                     if word not in self.stopwordset:
    #                         words_out.append(word)
    #                 f_out.write(' '.join(words_out) + '\n')
    #     print(321321)

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

    @staticmethod
    def get_ngrams(input, n):
        output = {}  # 构造字典
        for i in range(len(input) - n + 1):
            ngramTemp = "".join(input[i:i + n])  # .encode('utf-8')
            if ngramTemp not in output:  # 词频统计
                output[ngramTemp] = 0  # 典型的字典操作
            output[ngramTemp] += 1
        return output

    # 重新输出实体-长度，并排序,
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

    def init_ner(self):
        # f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt"
        f_in2 = "../data/nlpcc2016/result/e123.txt.statistics.txt"
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
                entity = str(entity)
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
    def clear_relations():
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clear.txt"
        f_out = f_in + ".alais_relations_1.txt"
        w_l = ['名', '称']
        w_2 = ['排', '第', '盛名', '专辑名称', '粉丝名称', '签名']

        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    l = ct.clean_str_rel(str(line).split('\t')[0])
                    if ct.end_with(l, w_l):
                        if not ct.contains_with(l, w_2):
                            out.write("%s\n" % l)

    # 实体重新抽取   ；实体别名
    # 红楼梦（中国古典长篇小说）
    # 《计算机基础》”，则将书名号内的部分作为该实体的别名
    # 如果抽取出的是唯一的，则是真正的别名
    @staticmethod
    def entity_re_extract():
        f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb_clean1_s.txt.statistics_len_and_sort.txt"
        f_out = f_in + ".alias_dict1.txt"
        d_dict = dict()
        p1 = '(\([^\((]*\))'
        p2 = '《[^《]*》'

        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
                for line in read_file:
                    l = line.split('\t')[0]
                    r1 = re.findall(p1, l)
                    r2 = re.findall(p2, l)

                    if len(r1) > 0 or len(r2) > 0:
                        list1 = list()
                        list1.append(l)
                        out.write("%s\t" % str(l))
                        for r in r1:
                            list1.append(r)
                            out.write("%s\t" % str(r).strip('《').strip('》')
                                      .strip('(').strip(')'))
                        for r in r2:
                            list1.append(r)
                            out.write("%s\t" % str(r).strip('《').strip('》')
                                      .strip('(').strip(')'))
                            # for l1 in list1:
                            #     out.write("%s\t" % str(l1).strip('《').strip('》')
                            #               .strip('(').strip(')'))
                            # print(str(l1))
                        out.write("\n")
                        #

        print(3421423)

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
    def combine_all_entitys():
        print(1111111111)
        # 读取1
        list1 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m1.txt')
        print(222222)
        # 读取2
        list2 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m2.txt')
        print(333333333)
        # 读取3
        list3 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m3.txt')
        # 合并输出
        print(0)
        s1 = set()
        for l in list1:
            for l1 in str(l).split('\t'):
                s1.add(l1)
        print(1)
        for l in list2:
            for l1 in str(l).split('\t'):
                s1.add(l1)
        print(2)
        for l in list3:
            # for l1 in str(l).split('\t'):
            s1.add(l)
        print(3)
        with codecs.open('../data/nlpcc2016/result/e123.txt', mode="w", encoding="utf-8") as out:
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
        self.list1 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m1.txt')
        print('e_by_m1')
        # 读取2
        self.list2 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m2.txt')
        print('e_by_m2')
        # 读取3
        self.list3 = ct.file_read_all_lines_strip('../data/nlpcc2016/result/e_by_m3.txt')
        print('e_by_m3')

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
            if find:
                origin_entitys.append(str(l).split('\t')[0])
                find = False
        print('1/3 %s' % (str(origin_entitys)))

        find = False
        for l in self.list2:
            for l1 in str(l).split('\t'):
                if l1 == dst:
                    find = True
            if find:
                origin_entitys.append(str(l).split('\t')[0])
                find = False
        print('2/3 %s' % (str(origin_entitys)))
        find = False
        for l in self.list3:
            if l == dst:
                find = True
            if find:
                origin_entitys.append(l)
                find = False
        print('3/3 %s' % (str(origin_entitys)))  # 按长度排序
        origin_entitys_no_repeat = []
        for e in origin_entitys:
            if e not in origin_entitys_no_repeat:
                origin_entitys_no_repeat.append(e)
        return origin_entitys_no_repeat

    def init_spo(self, f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt"):
        self.kbqa = dict()

        index = 0
        with codecs.open(f_in, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                index += 1
                if index % 10000 == 0:
                    print(index / 10000)
                ls = str(line).strip('\n').strip('\r').split('\t')
                s = ls[0]
                p = ct.clean_str_rel(ls[1])
                o1 = ls[2]
                t1 = (p, o1)
                if s in self.kbqa:
                    s1 = self.kbqa[s]
                    s1.add(t1)
                    self.kbqa[s] = s1
                else:
                    s1 = set()
                    s1.add(t1)
                    self.kbqa[s] = s1
        print("init_spo ok")
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
        e_s = self.kbqa.get(entity_id, "")
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


def method_name():
    bk = baike_helper()
    f_in = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt"
    f_out = f_in + "-out.txt"
    bk.convert_text_to_seg(f_in, f_out, type="questions")


def n_gram_math_all():
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
    bh.init_spo()
    ct.print_t(2)
    # bh.init_ner()
    bh.init_find_entity()
    ct.print_t(3)
    f_q_in = '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
    f_cand_q_in = '../data/nlpcc2016/extract_entitys.txt'
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
            o = ct.clean_str_rn(l.split('\t')[1])
            ss = str(cand_s[index]).split('\t')  # N-GRAM 实体名
            s_set = set()  # 候选实体集合
            ps = []
            es = []
            find_r = False
            for s1 in ss:
                if len(s1) == 1 or s1 in filter_words:
                    print(s1)
                    continue
                ps1 = bh.find_p(s1, o)
                if len(ps1) > 0:
                    ct.print_t("%s\t%s" % (s1, o))
                    es.append(s1)
                    ps.extend(ps1)
                    # 查找就停止
                    find_r = True
                    break
            find_r = True
            if find_r == False:
                for s1 in ss:
                    # 先遍历一遍 非1个单字的N-GRAM

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
                            "%s\t%s\t%s\t%d\t%d" % (es[0], ps[0][0], o, index, len(ps[0])))
            ct.print_t(ps)


if __name__ == '__main__':
    # baike_helper.e_r_combine()
    # method_name()
    # baike_helper.statistics_subject_len()
    # build_and_statistics_vocab()
    # s ="《机械设计基础》这本书的作者是谁"
    # method_name2()

    #
    # baike_helper.combine_all_entitys()
    # 2.5.4 按长度重写
    # f_in ='../data/nlpcc2016/result/e123.txt'
    # f_out= f_in+".statistics.txt"
    # baike_helper.statistics_subject_len(f_in,f_out)

    # 2.5.1 r_combine
    # baike_helper.r_combine()

    # 2.5.5 n-gram匹配
    # n_gram_math_all()

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

    find_all_ps_2_6_3()

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

    # 重写指定的实体

    print("finsih ")
