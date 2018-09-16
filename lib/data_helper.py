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
import math
import random

#20180906-1 用cos做NER



class DataClass:
    # ---------------------web questions
    relation_path = []  # 原始路径
    relation_path_clear = []  # 处理后的路径
    relation_path_one = []  # 处理后的只有随机的一个关系的路径
    relation_path_clear_str_all = []  # 处理后的路径集合，string格式
    # ---------------------freebase
    entitys = []
    relations = []
    relations_filter = []  # 待排除的关系 e.g. "type object creator"
    # ---------------------DataClass
    entity1_list = []  # id
    entity1_value_list = []  # 值

    entity1_in_q_list = []  # 实体在问句的候选中的部分
    entity1_in_q_cand_list = []  # 实体在问句的候选列表
    entity1_in_q_cand_list_origin = [] # 候选实体的原始

    relation_list = []  # 单词版
    entity2_list = []
    question_list = []
    question_list_index = []  # 数字索引版
    relation_list_index = []

    train_question_list_index = []  # 数字索引版
    train_relation_list_index = []

    test_question_list_index = []  # 数字索引版
    test_relation_list_index = []

    competing_train_score_dict = dict() # 竞争属性集合-训练集上的带分数
    fb = []
    # ----------------
    mode = ""  # 区分不同模式下函数的调用,可以考虑改成继承
    loss_ok = 0  # 记录loss=0的次数，停止

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
            # if self.mode=='cc':
            #     q_words_list = [x for x in sentence]
            # else:
            q_words_list = sentence.split(" ")
            for word in q_words_list:
                one_sentence.append(word)
            all_stence.append(one_sentence)
        return all_stence

    # 初始化待排除的关系，待考虑如何加进去
    def init_filter_relations(self):
        path = r"../data/freebase/filter_relations.txt"
        lines = ct.file_read_all_lines(path)
        lines = [str(x).replace("\n", "").replace("\r", "") for x in lines]
        return lines

    # run | init
    def __init__(self, mode="debug", run_type="run"):
        """
        mode = debug(1行数据调试);test(测试模式);small();

        :param mode:
        """
        # ---------------------初始化实体
        self.question_list_origin = []
        self.synonym_dict = dict()
        self.answer_list_index = []
        self.question_global_index = []
        self.question_labels = []  # 训练还是测试集的标记
        self.q_neg_r_tuple_train = []  # train
        self.q_neg_r_tuple_test = []  # test
        self.entity1_list = []
        self.relation_list = []
        self.entity2_list = []
        self.question_list = []
        self.entity_ner_list = []
        self.answer_list = []
        self.maybe_test_questions = []
        self.rdf_list = []
        self.mode = mode
        self.expend_es = []
        self.expend_score = []


        if mode == "cc":
            need_load_kb = True
            if need_load_kb:
                self.bh = baike_helper(config.cc_par('alias_dict'))
                self.bh.init_spo(f_in=config.cc_par('kb-use'))
                # 临时关闭

            self.init_cc_questions(config.cc_par('cc_q_path'), run_type)
            ct.print("init_cc_questions finish.")
            self.converter = read_utils.TextConverter(filename=config.par('cc_vocab'), type="zh-cn")
            if run_type == 'init':  # 初始化
                return
            msg = 'questions_len_train:%s\t wrong_relation_num:%s\t' % (
                config.get_static_q_num_debug(), config.get_static_num_debug())
            ct.print(msg, 'debug')

            self.load_all_q_r_tuple(config.get_static_q_num_debug(), config.get_static_num_debug(), is_record=True)
            self.get_max_length()
            self.q_r_2_arrary_and_padding()
            # 按比例分割训练和测试集
            self.division_data(0.8, config.cc_par('real_split_train_test'))
            self.build_embedding_weight(config.wiki_vector_path(mode))
            # 加载
            if config.cc_par('synonym_mode') == 'ps_synonym':
                self.init_synonym(config.cc_par('synonym_words'))
            if config.cc_compare('S_model', 'S_model'):
                self.synonym_train_data(config.cc_par('synonym_train_data'))
            # if config.cc_compare('pool_mode', 'competing_ps'):
            self.init_competing_model(config.cc_par('competing_ps_path'))
            # self.init_expend_es(config.cc_par('expend_es'))
            # 加载别名字典
            # self.init_alias_dict(config.cc_par('alias_dict'))
            ct.print("load embedding ok! start init nn")
            return
        # # elif mode == 'ner':
        # #     need_load_kb = True
        # #     if need_load_kb:
        # #         self.bh = baike_helper()
        # #         self.bh.init_spo(f_in=config.cc_par('kb-use'))
        # #
        # #     self.init_cc_questions(config.cc_par('cc_q_path'), run_type)
        # #     ct.print("init_cc_questions finish.")
        # #     self.build_vocab_ner()
        # #     # self.converter = read_utils.TextConverter(filename=config.par('cc_vocab'), type="zh-cn")
        # #     if run_type == 'init':  # 初始化
        # #         return
        # #     msg = 'questions_len_train:%s\t wrong_relation_num:%s\t' % (
        # #         config.get_static_q_num_debug(), config.get_static_num_debug())
        # #     ct.print(msg, 'debug')
        # #
        # #     self.load_all_q_r_tuple(config.get_static_q_num_debug(), config.get_static_num_debug(), is_record=True)
        # #     self.get_max_length()
        # #     self.q_r_2_arrary_and_padding()
        # #     # 按比例分割训练和测试集
        # #     self.division_data(0.8, config.cc_par('real_split_train_test'))
        # #     self.build_embedding_weight(config.wiki_vector_path(mode))
        #     # 加载
        #     # if config.cc_par('synonym_mode') == 'ps_synonym':
        #     #     self.init_synonym(config.cc_par('synonym_words'))
        #     # if config.cc_compare('S_model', 'S_model'):
        #     #     self.synonym_train_data(config.cc_par('synonym_train_data'))
        #     # if config.cc_compare('pool_mode', 'competing_ps'):
        #     #     self.init_competing_model(config.cc_par('competing_ps_path'))
        #
        # else:
        #     self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train-1.txt")
        #     self.init_fb("../data/freebase/fb_1000/")

        # # 建造词汇表
        # self.build_vocab()
        # # 获取最大长度
        # self.get_max_length()
        # # 问题和关系都转换成array形式，并padding问题
        # self.q_r_2_arrary_and_padding()
        # # 按比例分割训练和测试集
        # self.division_data()
        #
        # self.build_embedding_weight(config.wiki_vector_path(mode))
        # ct.print("load embedding ok!")
        #
        # self.build_all_q_r_tuple(config.get_static_q_num_debug(),
        #                          config.get_static_num_debug(), is_record=False)

        # self.load_all_q_r_tuple(config.get_static_q_num_debug(),
        #                         config.get_static_num_debug(), is_record=False)

        # ct.print("build_all_q_r_tuple 生成所有的q和neg r的组合")

    def q_r_2_arrary_and_padding(self):
        # 把问题和关系变成array形式
        self.question_list_split = self.get_split_list_per_line(self.question_list)
        self.relation_list_split = self.get_split_list_per_line(self.relation_list)
        self.answer_list_split = self.get_split_list_per_line(self.answer_list)
        for q_l_s in self.question_list_split:
            self.question_list_index.append(self.converter.text_to_arr_list(q_l_s))
        # self.relation_list_index = self.converter.text_to_arr(self.relation_list_split)
        for _ in self.relation_list_split:
            self.relation_list_index.append(self.converter.text_to_arr_list(_))
        for _ in self.answer_list_split:
            self.answer_list_index.append(self.converter.text_to_arr_list(_))

        # 第一版本先padding到max长度
        padding_num = self.get_padding_num()
        for index in range(0, len(self.question_list_index)):
            self.question_list_index[index] = \
                ct.padding_line(self.question_list_index[index], self.max_document_length, padding_num)
        for index in range(0, len(self.relation_list_index)):
            self.relation_list_index[index] = \
                ct.padding_line(self.relation_list_index[index], self.max_document_length, padding_num)
        for index in range(0, len(self.answer_list_index)):
            self.answer_list_index[index] = \
                ct.padding_line(self.answer_list_index[index], self.max_document_length, padding_num)

    def division_data(self, rate=0.8, real_split_train_test=False):
        ct.print("division_data start! real_split_train_test %s " % real_split_train_test, 'debug')
        # 6.2.2 选取指定个数的关系来实验
        # f1s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property'))
        # max_r_num = config.get_t_relation_num()
        # max_r_num = min(max_r_num, len(f1s))
        # f1s = f1s[0:max_r_num]


        # 在此对问题做一下设置，总问题数
        assert (len(self.question_list_index) == len(self.relation_list_index))
        max_q = config.get_total_questions()
        max_q = min(max_q, len(self.question_list_index))

        ct.print("division top %d for train and test(max_q=%d, total=%d) "
                 % (max_q, config.get_total_questions(), len(self.question_list_index)), 'debug')

        self.question_list_index = self.question_list_index[0:max_q]
        self.relation_list_index = self.relation_list_index[0:max_q]
        self.answer_list_index = self.answer_list_index[0:max_q]

        # self.question_global_index
        # self.train_question_global_index

        self.train_question_global_index = []
        self.test_question_global_index = []

        if real_split_train_test:
            skip = config.cc_par('real_split_train_test_skip')
            self.train_question_list_index, self.test_question_list_index, \
            self.train_question_global_index, self.test_question_global_index = \
                ct.cap_nums_by_skip(self.question_list_index, self.question_labels,
                                    skip, self.question_global_index)

            self.train_relation_list_index, self.test_relation_list_index, \
            self.train_question_global_index, self.test_question_global_index = \
                ct.cap_nums_by_skip(self.relation_list_index, self.question_labels,
                                    skip, self.question_global_index)

            self.train_answer_list_index, self.test_answer_list_index, \
            _1, _2 = \
                ct.cap_nums_by_skip(self.answer_list_index, self.question_labels,
                                    skip, self.question_global_index)

        else:
            self.train_question_list_index, self.test_question_list_index = \
                ct.cap_nums(self.question_list_index, rate)
            self.train_relation_list_index, self.test_relation_list_index = \
                ct.cap_nums(self.relation_list_index, rate)

        self.padding = 0  # 训练集和测试集合之间问题index的偏移量
        train_q_l = len(self.train_question_list_index)
        test_q_l = len(self.test_question_list_index)
        for qr_tuple in self.q_neg_r_tuple:
            if qr_tuple[0] < train_q_l:
                self.q_neg_r_tuple_train.append(qr_tuple)  # 问题是 (0- train_q_l)
            elif qr_tuple[0] < train_q_l + test_q_l:
                if self.padding == 0:
                    self.padding = qr_tuple[0]
                # 测试问题是 [train_q_l,train_q_l + test_q_l]
                self.q_neg_r_tuple_test.append(qr_tuple)
        if config.use_property():
            self.padding = train_q_l

        ct.print('use_property: %s' % config.use_property(), 'debug')
        ct.print(
            "division_data finish! train:%d test:%d" %
            (len(self.q_neg_r_tuple_train), len(self.q_neg_r_tuple_test)),
            'debug')

        # 记录问题集合和测试集合 输出问句
        ct.print('看看哪些neg关系是训练有', 'debug')
        r_train = set()
        r_test = set()
        ct.print('train', 'train_test_q')
        ct.print('\t%d\t%s\t%s\t%s\t%s' % (0, '实体', '关系', '问题', '被替换词'), 'train_test_q')
        for l in range(len(self.train_question_list_index)):
            ct.print("\t%d\t%d\t%s\t%s\t%s\t%s" % (
                self.question_global_index[l], l, self.entity1_list[l], self.relation_list[l],
                self.question_list[l], self.entity_ner_list[l]),
                     'train_test_q')
            r_train.add(self.relation_list[l])
        ct.print('test', 'train_test_q')
        for l in range(len(self.test_question_list_index)):
            global_index = l + self.padding
            ct.print("\t%d\t%d\t%s\t%s\t%s\t%s" % (self.question_global_index[global_index],
                                                   global_index, self.entity1_list[global_index],
                                                   self.relation_list[global_index],
                                                   self.question_list[global_index],
                                                   self.entity_ner_list[global_index]), 'train_test_q')
            r_test.add(self.relation_list[global_index])
        ct.print('test not in train', 'train_test_q')
        # 看看哪些pos关系是训练有，测试没有的
        r3 = (r_train | r_test) - r_train
        for r in r3:
            ct.print(r, 'train_test_q')

        # 看看哪些neg关系是训练有，测试没有的
        if False:
            ct.print('neg test not in train', 'train_test_q')
            neg_r_train = set()
            neg_r_test = set()
            for l in range(len(self.train_question_list_index)):
                global_index = l
                ps_to_except1 = self.relation_path_clear_str_all[global_index]
                rs, a_s = self.bh.read_entity_and_get_all_neg_relations_cc(self.entity1_list[global_index],
                                                                           ps_to_except1)
                for r in rs:
                    neg_r_train.add(r)

            for l in range(len(self.test_question_list_index)):
                global_index = l + self.padding
                ps_to_except1 = self.relation_path_clear_str_all[global_index]
                rs, a_s = self.bh.read_entity_and_get_all_neg_relations_cc(self.entity1_list[global_index],
                                                                           ps_to_except1)
                for r in rs:
                    neg_r_test.add(r)

            r4 = (neg_r_train | neg_r_test) - neg_r_train
            for r in r4:
                ct.print(r, 'train_test_q')

                # print('记录所有的训练和测试的负问句')
                # # 记录所有的训练和测试的负问句，其中测试负问句不会参与训练
                # # (len(self.q_neg_r_tuple_train), len(self.q_neg_r_tuple_test))
                # ct.print_list(["%s\t%s\t%s" % (x[0], x[1], x[2]) for x in self.q_neg_r_tuple_train], 'log_q_neg_r_tuple')
                # ct.print_list(['\nTEST\n'], 'log_q_neg_r_tuple')
                # ct.print_list(["%s\t%s\t%s" % (x[0], x[1], x[2]) for x in self.q_neg_r_tuple_test], 'log_q_neg_r_tuple')

    def get_max_length(self):
        if self.mode == "cc":
            max_document_length1 = max([len(x) for x in self.question_list])  # 获取单行的最大的长度
            max_document_length2 = max([len(x) for x in self.relation_list])  # 获取单行的最大的长度
        elif self.mode == "ner":
            max_document_length1 = max([len(x) for x in self.question_list])  # 获取单行的最大的长度
            max_document_length2 = max([len(x) for x in self.question_list])  # 获取单行的最大的长度
        else:
            # 将问题/关系转换成index的系列表示
            max_document_length1 = max([len(x.split(" ")) for x in self.question_list_origin])  # 获取单行的最大的长度
            # max_document_length1 = max([len(x.split(" ")) for x in self.question_list])  # 获取单行的最大的长度
            max_document_length2 = max([len(x.split(" ")) for x in self.relation_list])  # 获取单行的最大的长度
        # gth = []
        # for x in self.relation_path_clear_str_all:
        #     for x1 in x:
        #         gth.append(len(x1.split(" ")))
        # max_document_length3 = max(gth)  # 获取单行的最大的长度
        # 计算出平均的长度
        # mean_of_quesitons = np.mean([len(x.split(" ")) for x in self.question_list])
        # ct.print(mean_of_quesitons)
        self.max_document_length = max(max_document_length1, max_document_length2)
        ct.print("q:%s r:%s" % (max_document_length1, max_document_length2), "debug")

    def build_vocab(self):
        # 建造词汇表
        # 将问题和关系的字符串变成以空格隔开的一个单词的list
        # total_list = self.question_list + self.relation_list
        q_words = self.get_split_list(self.question_list)
        q_words.extend(self.get_split_list(self.relations))  # freebase里面的关系
        # 应该再加上问题里面的关系集合
        # q_words = [str(x).replace(".","") for x in q_words ]
        self.converter = read_utils.TextConverter(q_words)
        # self.converter.save_to_file_raw(
        #     log_path + "/vocab_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + str(".txt"))
        # ct.print("save_to_file_raw ok!")
        # self.converter.save_to_file("model/converter.pkl")
        # ct.print(self.converter)

    def build_vocab_cc(self):
        # 建造词汇表
        # 将问题和关系的字符串变成以空格隔开的一个单词的list
        # total_list = self.question_list + self.relation_list
        q_words = self.get_split_list(self.question_list)
        q_words.extend(self.get_split_list(self.relations))  # freebase里面的关系
        # 应该再加上问题里面的关系集合
        # q_words = [str(x).replace(".","") for x in q_words ]
        self.converter = read_utils.TextConverter(q_words)
        # self.converter.save_to_file_raw(
        #     log_path + "/vocab_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + str(".txt"))
        # ct.print("save_to_file_raw ok!")
        # self.converter.save_to_file("model/converter.pkl")
        # ct.print(self.converter)

    # ---------------------simple questions

    def init_simple_questions(self, file_name):
        # line_list = []
        idx = 0
        # www.freebase.com/m/04whkz5	www.freebase.com/book/written_work/subjects
        # www.freebase.com/m/01cj3p
        # what is the book e about
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file.readlines():
                    idx += 1
                    line_seg = line.split('\t')
                    # www.freebase.com/m/04whkz5
                    entity1 = line_seg[0].split('/')[2]
                    relation1 = ct.clear_relation(line_seg[1])

                    entity2 = line_seg[2].split('/')[2]
                    question = ct.clear_question(line_seg[3])

                    self.entity1_list.append(entity1)
                    self.relation_list.append(relation1)
                    # self.entity1_list.append(entity2)
                    self.question_list.append(question)
                    self.relation_path_clear_str_all.append([relation1])
                    # self.rdf_list.append([entity1, relation1, entity2])
                    # check it
                    # line_list.append(line)
            except Exception as e:
                ct.print("index = ", idx)
                logging.error("error ", e)
        ct.print("entity1_list:%d " % len(self.entity1_list))

    # -----------------cc 将问答集合，
    def init_cc_questions(self, file_name, run_type='run'):
        f1s_new = []
        idx = -1
        # 《机械设计基础》这本书的作者是谁？    杨可桢，程光蕴，李仲生
        # 机械设计基础         作者          杨可桢，程光蕴，李仲生
        # 问题0 答案1 实体s-2 关系p-3 属性值o-4    匹配到的实体s-5
        # 读取问答全体数据集

        bad_idx = []
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file.readlines():
                    idx += 1
                    line_seg = line.split('\t')
                    if len(line_seg) < 6 or line.__contains__('NULL'):  #
                        ct.print("bad:" + line, "bad")
                        bad_idx.append(idx) # 记录需要跳过的index
                        continue
                    f1s_new.append(line)
            except Exception as e:
                print(e)
                ct.print("index = ", idx)
                logging.error("error ", e)



        use_property = config.use_property()
        property_list = []

        if run_type == 'init':
            use_property = 'none'
            f2s = []

        if use_property == 'num':
            f2s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property'))
            f2s = ct.list_safe_sub(f2s, config.get_t_relation_num())
        elif use_property == 'special':
            f2s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property'))
            # f3s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property_test'))
            # # 选出所有的实体，筛选一遍f2s
            # f3s = [str(x).split('\t')[0] for x in f3s]
            # f2s_new = []
            # for x in f2s:
            #     if str(x).split('\t')[0] in f3s:
            #         f2s_new.append(x)
            # f2s = f2s_new
            f2s = ct.list_safe_sub(f2s, config.get_t_relation_num())
        elif use_property == 'maybe':
            f2s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property'))
            # f3s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property_test'))
            f4s = ct.file_read_all_lines_strip(config.cc_par('rdf_maybe_property'))
            f5s = ct.file_read_all_lines_strip(config.cc_par('rdf_maybe_property_index'))
            index = int(f5s[len(f5s) - 1])  # 取最后一行
            # 筛选1遍f2s
            if index > len(f4s):
                raise Exception('完成')
            for i in range(len(f4s)):
                if i < index:
                    continue
                if str(f4s[i]).__contains__('===='):
                    continue
                index = i
                # test_id = int(str(f4s[i]).split('\t')[1])
                test_ids = []
                if str(f4s[i]).split('\t')[1].__contains__('_'):
                    test_ids = [x for x in str(f4s[i]).split('\t')[1].split('_')]
                    test_ids_str = '\t'.join(test_ids)
                else:
                    test_ids.append(int(str(f4s[i]).split('\t')[1]))
                    test_ids_str = str(f4s[i]).split('\t')[1]
                self.maybe_test_questions = test_ids
                ct.print('test_id = %s' % test_ids_str, 'train_test_q')
                f3s = str(f4s[i]).split('\t')[2:]
                break

            ct.just_log(config.cc_par('rdf_maybe_property_index'), str(index + 1))
            # +1 然后处理下一个

            # 选出所有的实体，筛选一遍f2s
            # f3s = [str(x).split('\t')[0] for x in f3s]
            f2s_new = []
            for x in f2s:
                if str(x).split('\t')[0] in f3s:
                    f2s_new.append(x)
            f2s = f2s_new
            f2s = ct.list_safe_sub(f2s, config.get_t_relation_num())
        else:
            # raise Exception('error use_property')
            ct.print("use_property=%s,no的时候只适用于生成模式非训练模型 " % use_property)

        for f2s_line in f2s:
            property_list.extend(str(f2s_line).split('\t')[2:])
            ct.print(str(f2s_line).split('\t')[0], 'use_r')
        property_list = [int(x) for x in property_list]

        index = -1
        for line in f1s_new:
            index += 1
            if (use_property in ['num', 'special', 'maybe']) and index not in property_list:
                continue
            line_seg = line.replace('\r', '').replace('\n', '').split('\t')
            answer = line_seg[1]
            entity1 = line_seg[2]
            relation1 = ct.clean_str_rel(line_seg[3].lower())  # 清洗关系
            entity2 = line_seg[4]
            entity_ner = line_seg[5].replace('\n', '').replace('\r', '')

            # 6.1.1.3 3 在载入问题的时候用♠替换掉实体
            # question = line_seg[0]
            # question = question.replace(' ', '').lower()
            # if not question.__contains__(entity_ner):
            #     ct.print(question, 'entity_ner')
            # question = question.replace(entity_ner, '♠')
            # 改成直接取第7列
            # 如果是测试 之后使用question的时候自己转换
            # if config.cc_compare('train_part','relation'):
            question = line_seg[6]
            # elif config.cc_compare('train_part','entity'):
            #     question = line_seg[0] #
            question_origin = str(line_seg[0]).replace(' ', '').lower()

            #
            real_entity_index = line_seg[8]
            cand_entity = line_seg[9:]
            if real_entity_index == '-1':
                # f1s_remove_item = str(l1).split('\t')[1:][f1s_index]
                # if len(self.entity1_list) > idx2:
                self.entity1_in_q_list.append(entity1)  # 不存在的时候用实体替代
                # else:
                #     self.entity1_in_q_list.append('####')  # 不存在的时候用实体替代
                self.entity1_in_q_cand_list.append(cand_entity)
                self.entity1_in_q_cand_list_origin.append(cand_entity)
            else:
                real_entity_index = int(real_entity_index)
                f1s_remove_item = cand_entity[real_entity_index]
                self.entity1_in_q_list.append(f1s_remove_item)
                _1 = cand_entity.copy()
                _1.remove(f1s_remove_item)
                self.entity1_in_q_cand_list.append(_1)
                self.entity1_in_q_cand_list_origin.append(cand_entity)

            #
            self.entity1_list.append(entity1)
            self.relation_list.append(relation1)
            self.entity2_list.append(entity2)

            self.question_list.append(question)  # 将问题替换掉
            self.question_list_origin.append(question_origin)  # 将问题替换掉
            self.entity_ner_list.append(entity_ner)
            self.answer_list.append(answer)

            rs1 = [relation1]
            vs = self.bh.kbqa.get(entity1, '')
            if vs != '':
                for k, v in vs:
                    if ct.padding_answer(v) == ct.padding_answer(answer):
                        if k not in rs1:
                            rs1.append(k)
                            # except Exception as ee1:
                            #     ct.print(ee1)
            self.relation_path_clear_str_all.append(rs1)

            # 增加一个容器 标记所有的问题是否属于训练集合还是测试集合
            is_train = index > config.cc_par('real_split_train_test_skip')
            self.question_labels.append(is_train)

            self.question_global_index.append(index)

            # 加载扩展的实体集合
            # if False:
            #     vs = line_seg[8:]
            #     _index = -1
            #     # d1 = dict()
            #     list1 = []
            #     list2 = []
            #     s1 = '____'
            #     for vs1 in vs:
            #         _index += 1
            #         # d1[str(index)] = vs1[1:] # KEY=第几个，value 对应的实体
            #         vs2 = str(vs1).split(s1)
            #         #  截取分数之后的
            #         list2.append(vs2[1])
            #         list1.append(vs2[2:]) # 不加所有 只加1个
            #         # list1.append([entity1])  # 不加所有 只加1个
            #     self.expend_score.append(list2)
            #     self.expend_es.append(list1)


        # 20180906-1 增加读取NER的结果---start
        # f1s = ct.file_read_all_lines_strip(config.cc_par('ner_path'))
        # idx2 = -1
        # for l1 in f1s:
        #     idx2 += 1
        #     #     if idx2 in bad_idx:
        #     #         continue # 如果是上面要求跳过的，则这里也跳过
        #     # 已经跳过了
        #     f1s_index = str(l1).split('\t')[0]
        #     # 对于没有识别出的，先不管他
        #     if f1s_index == '-1':
        #         # f1s_remove_item = str(l1).split('\t')[1:][f1s_index]
        #         if len(self.entity1_list)>idx2:
        #             self.entity1_in_q_list.append(self.entity1_list[idx2])  # 不存在的时候用实体替代
        #         else:
        #             self.entity1_in_q_list.append('####') # 不存在的时候用实体替代
        #         self.entity1_in_q_cand_list.append(str(l1).split('\t')[1:])
        #     else:
        #         f1s_index = int(f1s_index)
        #         f1s_remove_item = str(l1).split('\t')[1:][f1s_index]
        #         self.entity1_in_q_list.append(f1s_remove_item)
        #         _1 = str(l1).split('\t')[1:]
        #         _1.remove(f1s_remove_item)
        #         self.entity1_in_q_cand_list.append(_1)
        # 20180906-1 增加读取NER的结果---end

        ct.print("entity1_list:%d " % len(self.entity1_list))
        if len(self.entity1_list) == 0:
            raise Exception('entity1_list 长度为0 file_name= %s ' % file_name)

    # brazil	/m/015fr@@1~/m/03385m^/location/country/currency_used@@1~text
    # Brazilian real	what type of money does brazil have?
    def init_web_questions(self, fname=r'../data/web_questions/rdf.txt'):
        total_useless = 0
        index = -1
        with codecs.open(fname, mode='r', encoding='utf-8') as read_file:
            for line in read_file.readlines():
                # line = f1.readline()
                line = ct.clean_str_simple(line)
                entity1 = line.split('\t')[0]
                relation_path = line.split('\t')[1]
                # answer = line.split('\t')[2]
                question = line.split('\t')[3]

                if relation_path.__contains__("###"):
                    # self.relation_path_clear.append(["###"])
                    # self.relation_list.append("###")
                    total_useless += 0
                    continue

                index += 1
                self.entity1_list.append(entity1)
                self.relation_path.append(relation_path)
                self.question_list.append(question.replace("\n", ""))

                # 解出1行所有正确的关系，
                relation_path_rs_all = ct.decode_all_relations(relation_path)
                # 随机获取1行作为他的关系
                one_relation = ct.random_get_one_from_list(relation_path_rs_all)

                self.relation_path_one.append(one_relation)
                # 展开一个关系，加入
                temp_relation = ""
                for o_r in one_relation:
                    temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
                #
                self.relation_list.append(temp_relation)

                # 处理一下添加到这里
                # self.relation_list 这个格式是 空格隔开的单词
                self.relation_path_clear.append(relation_path_rs_all)

                # 输出
                # msg = "%d\t%s\t"%(index,entity1)

                # 将处理后的路径集合，转换成string格式加入relation_path_clear_str_all
                relation_path_rs_str_all = []
                for x in relation_path_rs_all:
                    temp_relation = ""
                    for o_r in x:
                        temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
                    # 增加关系 ' tv tv actor starring roles  tv regular tv appearance character text '
                    relation_path_rs_str_all.append(temp_relation)
                # to do: here to record r path 在这 记录关系路径
                # msg = msg + "^".join(relation_path_rs_str_all)
                # ct.just_log("../data/web_questions/q_relations_path.txt",msg )
                self.relation_path_clear_str_all.append(relation_path_rs_str_all)

            ct.print("end total_useless = %d " % total_useless)

    # ---------------------freebase
    def init_fb(self, file_name="../data/freebase/"):
        # file_name1 = "freebase_entity.txt"
        # file_name2 = "freebase_rdf.txt"
        file_name3 = "freebase_relation_clear.txt"  # 所有的关系
        # 装载entity_id
        # with codecs.open(file_name + file_name1, mode="r", encoding="utf-8") as read_file:
        #     for line in read_file.readlines():
        #         self.entitys.append(line.replace("\n", "").replace("/m/", "").replace("\r", ""))
        # ct.print("entitys len:" + str(len(self.entitys)))
        # 装载freebase的关系
        with codecs.open(file_name + file_name3, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relations.append(
                    line.replace("\n", "").replace("/", " ").replace(".", " ").replace("_", " ").strip())
        ct.print("relations len:" + str(len(self.relations)))
        # relation_path_clear_str_all

    def init_relation_fb(self, file_name="../data/freebase/freebase_relation_clear.txt"):
        """
        从文件中加载所有的关系然后作为词汇的候选列表
        :return:
        """
        # self.relation_list
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relation_list.append(line.replace("\r\n", ""))
                # ct.print("init_relation_fb")

    def get_relations_except_one(self, path, entity_id, ps_to_except):
        """
        获取除了指定关系外的所有关系
        :param entity_id:
        :param ps:
        :return:
        """
        id, ps = self.read_rdf_from_gzip_or_alias(path, entity_id)
        if id == "":
            return []
        ps_to_return = []
        for p in ps:
            if p not in ps_to_except:
                ps_to_return.append(p)
        return ps_to_return

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
                ct.print(e1, id)
        return exist

    # --------------------生成batch 没用
    def batch_iter_wq(self, question_list_index, relation_list_index, batch_size=100):
        ct.print("enter:batch_iter_wq")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本

        self.shuffle_indices_train = shuffle_indices[0:batch_size]  # 取出指定的样本记录下来
        msg1 = "shuffle_indices q= %s " % str(self.shuffle_indices_train)
        ct.just_log2("info", msg1)
        ct.print(msg1, "data")

        total = 0
        for index in shuffle_indices:
            x_new.append(x[index])
            y_new.append(y[index])

            question = self.question_list[index]
            name = self.entity1_list[index]

            #  得到该id所有的正确和错误的关系
            ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
            ps_to_except1 = [ps_to_except1]
            # 选出neg关系添加
            r1, r1_index = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            r1_text = r1
            r1 = self.converter.text_to_arr_list(r1)
            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())
            z_new.append(r1)

            # log
            info1 = "%d q:%s e:%s  %d,%d" % (index, question, name, len(ps_to_except1), len(r_all_neg))
            ct.print(info1)
            ct.just_log2("info", info1)

            msg = "qid = %d,neg r=%d r=%s " % (index, r1_index, r1_text)
            ct.log3(msg)

            msg_right = "r-pos %d :%s       " % (len(str(ps_to_except1[0]).split(" ")), ps_to_except1[0])
            ct.just_log2("info", msg_right)
            ct.print(msg_right)

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            ct.just_log2("info", msg_neg)
            ct.print(msg_neg)

            #
            total += 1
            if total >= batch_size:
                break
                # total = 0
                # yield np.array(x_new), np.array(y_new), np.array(z_new)

        ct.print(shuffle_indices[0:batch_size])
        # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
        # rate = 1
        # r_si = reversed(shuffle_indices)
        # r_si = list(r_si)
        # ct.print(r_si)
        # total = 0
        # for index in r_si:
        #     z_new.append(y[index])
        #     total += 1
        #     if total >= batch_size:
        #         break
        # ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # -------------------测试生成同一批次的 debug 在测！！！
    # 在用
    def batch_iter_wq_debug(self, question_list_index, relation_list_index, shuffle_indices, batch_size, train_part):
        """
        web questions 的生成反例的办法。debug版本，
        生成指定batch_size的数据。

        update:
        1. 改为1个epoches获取不到重复的数据.--2018年1月23日11:56:22

        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_debug")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        # y_a = answer_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        # z_a_new = []
        # self.q_neg_r_tuple 这个地方需要筛选出仅有问题列表里面的数据


        # 生成 0- len(question_list_index) 的随机数字
        total = len(self.q_neg_r_tuple_train)
        if len(shuffle_indices) == 0:
            shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标

        info1 = "q total:%d ; epohches-size:%s " % (total, len(self.q_neg_r_tuple_train) / batch_size)
        ct.print(info1, 'info')

        for list_index in range(total):
            # 获取q_neg_r_tuple里面打乱的下标的对应的 q_r 对
            try:
                q_neg_r = self.q_neg_r_tuple_train[shuffle_indices[list_index]]
            except Exception as e1:
                print(e1)
            index = q_neg_r[0]  # 对应类里面的index
            name = q_neg_r[1]  # 问题
            r_neg = q_neg_r[2]  # 关系
            a_neg = q_neg_r[3]  # 答案

            if train_part == 'relation':
                train_part = r_neg
            else:
                train_part = a_neg
            train_part = [train_part]

            x_new.append(x[index])  # 添加问题
            y_new.append(y[index])  # 添加正确的关系
            ct.print(x[index], "debug_epoches")
            ct.print(y[index], "debug_epoches")

            r1 = self.converter.text_to_arr_list(train_part)  # 文字转数字
            ct.print(r1, "debug_epoches")
            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())
            z_new.append(r1)  # 添加错误的关系

            # log
            info1 = "q=%d ,r-pos=%d,r-neg=%d q=%s e=%s  " % (
                index, index, list_index,
                self.converter.arr_to_text_no_unk(x[index]).replace('<unk>', '').replace(' ', ''),
                self.entity1_list[index])
            ct.print(info1[0:30], "debug")
            ct.just_log2("info", info1)
            msg = "qid=%d,neg r=%d  " % (index, list_index)
            ct.log3(msg)
            msg_neg = "r-neg %d :%s       " % (list_index, train_part)
            ct.just_log2("info", msg_neg)

            if list_index % batch_size == 0 and list_index != 0:
                x_return = x_new.copy()
                y_return = y_new.copy()
                z_return = z_new.copy()
                x_new.clear()
                y_new.clear()
                z_new.clear()
                yield np.array(x_return), np.array(y_return), np.array(z_return)

    def batch_iter_wq_debug_fix_model(self, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list,
                                      batch_size=100):
        x_new = []
        y_new = []
        z_new = []
        batch_size = min(batch_size, len(error_test_q_list))
        shuffle_indices = np.random.permutation(np.arange(len(error_test_q_list)))  # 打乱样本下标
        for list_index in range(len(shuffle_indices)):
            try:
                x_new.append(error_test_q_list[shuffle_indices[list_index]])
                y_new.append(error_test_pos_r_list[shuffle_indices[list_index]])
                z_new.append(error_test_neg_r_list[shuffle_indices[list_index]])
            except Exception as e1:
                print(e1)

            ct.print("list_index:" + str(list_index), "debug")
            # if list_index == 0: # 默认是大于0的一个size
            #     continue
            if (list_index + 1) % batch_size == 0:
                x_return = x_new.copy()
                y_return = y_new.copy()
                z_return = z_new.copy()
                x_new.clear()
                y_new.clear()
                z_new.clear()
                yield x_return, y_return, z_return

    # 在用，valid_batch_debug-> 生成一个问题的相关信息
    def batch_iter_wq_test_one_debug(self,  model, index,train_part='relation'):

        ct.print("enter:batch_iter_wq_test_one_debug")

        x_new = []  # 问题集合
        y_new = []  # 关系集合
        y_a_new = []  # 答案集合
        z_new = []  #
        labels = []  # 标签集合
        synonym_score = []  # 每个分数

        if model == "valid":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
        else:
            raise Exception("MODEL 参数出错")
        if global_index >= len(self.entity1_list):
            ct.print('error ! ', 'error')

        # log
        ct.just_log2("info", "\nbatch_iter_wq_test_one_debug=================================start")
        msg = "%s\t%s\t%d\t%d" % (
            model, index, global_index, self.question_global_index[global_index])
        ct.print(msg, 'debug')
        ct.log3(msg)
        ct.just_log2("info", msg)
        part1 = msg

        name = self.entity1_list[global_index]
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的

        if train_part == 'relation':
            rs, a_s = self.bh.read_entity_and_get_all_neg_relations_cc(entity_id=name, ps_to_except=ps_to_except1)
        elif train_part == 'entity':
            cand_s = self.entity1_in_q_cand_list[global_index]
            rs = self.bh.rs_cc_subject(cand_s, 100)

        # 添加正确答案
        #  增加synonym模式
        x_new_current = None # 当前模式下的
        if config.cc_par('synonym_mode') == 'ps_synonym':
            # 计算每个属性的可扩展范围
            # 1 pos属性的
            r_pos = self.relation_list[global_index]
            r_neg_list = rs
            r_all = []
            r_all.append(r_pos)
            r_all.extend(r_neg_list)

            s_dict = ct.dict_get_synonym(self.synonym_dict, r_all)
            # 校验
            # for _ in r_all:
            #     s1 = s_dict[_]
            #     # ps_sorted = ct.sort_synonym_ps(s1, q, 5)
            #     for _1 in s1:
            #         print("%s\t%s" % (_1[0], _1[1]))
            #     print('-----')

            # 获取指定属性的扩展属性
            r_pos = self.relation_list[global_index]
            _ps = s_dict.get(r_pos)
            # 对PS集合做一下过滤
            q = self.question_list[global_index]
            _ps = ct.sort_synonym_ps(_ps, q, r_pos)
            for _ps_tuple in _ps:
                _ps_item = _ps_tuple[0]
                x_new.append(self.question_list_index[global_index])
                y_new.append(self.convert_str_to_indexlist(_ps_item))
                labels.append(True)
                _k1 = "%s\t%s" % (r_pos, _ps_item)
                _v1 = self.synonym_score_dict.get(_k1, '-1')
                # 原始属性，当前属性，当前属性得分，是否原本属性，该属性的字表面得分
                synonym_score.append(
                    (r_pos, _ps_item, str(_v1), str(r_pos == _ps_item), str(_ps_tuple[1])))  # r_pos==_ps_item 表示是否自己
        else:
            if train_part == 'relation':
                x_new_current = self.question_list_index[global_index]

                y_new.append(self.relation_list_index[global_index])
            elif train_part == 'entity':  # 20180906-1 用cos做NER
                ner_q = self.question_list_origin[global_index]  #
                x_new_current = self.convert_str_to_indexlist(ner_q)
                y_new.append(self.convert_str_to_indexlist(self.entity1_in_q_list[global_index]))

            x_new.append(x_new_current)  #
            labels.append(True)

        ct.just_log2("info", "entity:%s " % name)
        part4 = "%s " % name

        # ct.print(y[index])
        if train_part == 'relation':
            r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        elif train_part == 'entity':
            r1_text = self.entity1_in_q_list[global_index]

        q1_text = self.converter.arr_to_text_no_unk(x_new_current)
        r1_msg = "r/e-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)
        part2 = q1_text
        part3 = "%s\t%s" % (r1_text, self.answer_list[global_index])
        ct.just_log3("test_check", "%s\t%s\t%s\t%s\t" % (part1, part2, part4, part3))

        # if train_part == 'relation':
        #     rs = rs
        # else:
        #     rs = a_s
        rs_len = len(rs)
        num = min(ct.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        for r1 in rs:
            # r1_split = [r1]  # .split(" ")
            # r1 = self.converter.text_to_arr_list(r1_split)
            # # r1_text = self.converter.arr_to_text_no_unk(r1)
            # # ct.log3(r1_text)
            # # ct.just_log2("info", "r1_neg in test %s" % r1_text)
            # # ct.print(r1_text)
            # # ct.just_log2("info","neg-r test:" + r1_text)
            # r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            if config.cc_par('synonym_mode') == 'ps_synonym':
                _rs = s_dict.get(r1)
                # 对PS集合做一下过滤
                q = self.question_list[global_index]
                _rs = ct.sort_synonym_ps(_rs, q, r1)
                for _ps_tuple in _rs:
                    _ps_item = _ps_tuple[0]
                    r1_index_list = self.convert_str_to_indexlist(_ps_item)
                    x_new.append(self.question_list_index[global_index])
                    y_new.append(r1_index_list)  # neg
                    labels.append(False)
                    _k1 = "%s\t%s" % (r1, _ps_item)
                    _v1 = self.synonym_score_dict.get(_k1, '-1')
                    # 原始属性，当前属性，当前属性得分，是否原本属性，该属性的字表面得分
                    # synonym_score.append((_ps_item, _ps_item, _v1, str(r_pos == _ps_item),_ps_tuple[1]))
                    synonym_score.append((r1, _ps_item, str(_v1), str(r1 == _ps_item), str(_ps_tuple[1])))
            else:
                x_new.append(x_new_current)
                y_new.append(self.convert_str_to_indexlist(r1))  # neg
                labels.append(False)

        # ct.print("show shuffle_indices")
        # ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
        ct.print("leave:batch_iter_wq_test_one_debug")

        if config.cc_par('synonym_mode') == 'ps_synonym':
            return np.array(x_new), np.array(y_new), [labels, synonym_score]
        else:
            return np.array(x_new), np.array(y_new), labels


    # batch_iter_cc_answer_test_one_debug
    # 答案选择模块的产生训练或者测试的问题
    def batch_iter_cc_answer_test_one_debug(self, question_list_index, relation_list_index, model, index,
                                            train_part='relation'):
        """
        web questions
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_cc_answer_test_one_debug")

        x_new = []  # 问题集合
        y_new = []  # 关系集合
        y_a_new = []  # 答案集合
        z_new = []  #
        labels = []  # 标签集合
        synonym_score = []  # 每个分数
        es_name_labels = []
        ner_score_list = []

        if model == "valid":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
        else:
            raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_wq_test_one_debug=================================start")
        msg = "%s\t%s\t%d\t%d" % (
            model, index, global_index, self.question_global_index[global_index])
        ct.print(msg, 'debug')
        ct.log3(msg)
        ct.just_log2("info", msg)
        part1 = msg

        if global_index >= len(self.entity1_list):
            ct.print('error ! ', 'error')
        # 获取实体名
        es_all = []
        try:
            for _item in self.expend_es[global_index]:
                if len(_item) != 0:
                    es_all.extend(_item)
        except Exception as e1:
            print(e1)
        # es_all = list(set(es_all))
        # 在这里加入分数


        name = self.entity1_list[global_index]

        # ps_to_except1 = self.relation_list[global_index]  # 应该从另一个关系集合获取
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的
        # ps_to_except1 数组组合

        answer_to_except = self.answer_list[global_index]
        # 实体-属性-属性值-是否是正确的属性
        # all_tuple = self.bh.answer_get_all_neg_relations_cc(es_all, ps_to_except1,score1)

        all_tuple = []
        top_k = [1, 2, 3]
        for k in top_k:
            score1 = self.expend_es[global_index][0]  # score int
            try:
                tuple1 = self.bh.answer_get_all_neg_relations_cc(self.expend_es[global_index][k - 1], ps_to_except1,
                                                                 score1)
                all_tuple.extend(tuple1)
            except Exception as e1:
                print(e1)


        ct.just_log2("info", "entity:%s " % name)
        part4 = "%s " % name

        # ct.print(y[index])
        r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        q1_text = self.converter.arr_to_text_no_unk(self.question_list_index[global_index])
        r1_msg = "r-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)
        part2 = q1_text
        part3 = "%s\t%s" % (r1_text, self.answer_list[global_index])
        ct.just_log3("test_check", "%s\t%s\t%s\t%s\t" % (part1, part2, part4, part3))

        for index in range(len(all_tuple)):  # 遍历all_tuple 设定数据
            match_s = all_tuple[index][0]
            p = all_tuple[index][1]
            o = all_tuple[index][2]
            is_right = all_tuple[index][3]
            entity_name_in_kb = all_tuple[index][4]
            ner_score = all_tuple[index][5]
            q = self.question_list_origin[global_index]
            q = str(q).replace(match_s, '♠')
            x_new.append(self.convert_str_to_indexlist(q))
            y_new.append(self.convert_str_to_indexlist(p))  # neg
            labels.append(is_right)
            es_name_labels.append(entity_name_in_kb)
            ner_score_list.append(ner_score)

        ct.print("leave:batch_iter_cc_answer_test_one_debug")
        if len(x_new) == 0:
            print(111)
        return np.array(x_new), np.array(y_new), [labels, es_name_labels, ner_score_list]

    # --------------------按比例分割
    def cap_nums(self, y, rate=0.8):
        y = y.copy()
        y = np.array(y)
        s = 0
        total_len = len(y)
        total_index = total_len * rate
        e = int(total_index)
        reverseIndex = int(total_len - total_index)
        y1 = y[s:e]  # [ > s and <= e  ]
        y2 = y[-reverseIndex:]
        ct.print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
        return y1, y2

    def just_log(self, file_name, msg):
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()
        # ct.print(1)

    # -------------------init web questions
    def add_relation_path_rs(self, relation_path_rs):
        rs = []
        for i in relation_path_rs:
            rs.append(i)
        return rs

    # ---------------------------------零碎的小东西
    def get_padding_num(self):
        return self.converter.vocab_size - 1

    # --------------------------------构建word2vec的矩阵
    def build_embedding_weight(self, filename, embedding_size=100):
        """
        load embedding
        """
        embeddings = []
        word2idx = dict()
        idx2word = dict()
        idx = 0
        with codecs.open(filename, mode="r", encoding="utf-8") as rf:
            try:
                for line in rf.readlines():
                    idx += 1
                    arr = line.split(" ")
                    if len(arr) != (embedding_size + 1):
                        logging.error("embedding error, index is:%s" % (idx))
                        continue

                    embedding = [float(val) for val in arr[1:]]  # 整行过去
                    word2idx[arr[0]] = len(word2idx)
                    idx2word[len(word2idx)] = arr[0]
                    embeddings.append(embedding)

            except Exception as e:
                logging.error("load embedding Exception,", e)
            finally:
                rf.close()

        logging.info("load embedding finish!")
        self.embeddings = embeddings  # np.ndarray(embeddings)
        self.word2idx = word2idx
        self.idx2word = idx2word
        return embeddings, word2idx, idx2word

    # def prodeuce_embedding_vec_file(self, filename):
    #     model = models.Word2Vec.load(filename)
    #     # 遍历每个单词，查出word2vec然后输出
    #
    #     v_base = model['end']
    #     ct.print(v_base)
    #
    #     for word in self.converter.vocab:
    #         try:
    #             v = model[word]
    #         except Exception as e1:
    #             msg1 = "%s : %s " % (word, e1)
    #             ct.print(msg1)
    #             ct.just_log("../data/word2vec/wiki.vector2.log", msg1)
    #             v = model['end']
    #         m_v = ' '.join([str(x) for x in list(v)])
    #         msg = "%s %s" % (word, str(m_v))
    #         # ct.print(msg)
    #         ct.just_log("../data/word2vec/wiki.vector2", msg)
    #     msg = "%s %s" % ('end', str(v_base))
    #     ct.just_log("../data/word2vec/wiki.vector2", msg)





    # 生成一个epoches中的一个batch
    def build_all_q_r_tuple(self, questions_len_train, error_relation_num=9999, is_record=False):
        """
        V1.0 根据实体的NEG属性每条都生成一条训练数据。
        V2.0
            1.  扫一遍扫描测试集，收集完所有的属性(ps_set)，
            2.  在训练集中找到所有的他们作为POS的问句(q_set)，
            3.  针对每个问句，如果(ps_set)中的每个属性不是POS则是NEG
        :param questions_len_train:
        :param error_relation_num:
        :param is_record:
        :return:
        """
        # 组合所有的问题和错误关系放进一个tuple中
        # self.question_list
        # self.relation_path_clear_str_all 正确关系
        build_version = 1
        # 1.  扫一遍扫描测试集，收集完所有的属性(ps_set)，
        # self.question_labels
        neg_ps_set = set()
        neg_ps_dict = dict()
        if build_version == 2:
            for index in range(len(self.question_labels)):
                if self.question_labels[index]:
                    s1 = self.entity1_list[index]
                    # ps_to_except1 = self.relation_path_clear_str_all[index]
                    vs1 = self.bh.kbqa.get(s1, "")
                    if vs1 == '':
                        continue
                    for _vs1 in vs1:
                        if _vs1 not in neg_ps_dict:
                            _tmp_set = set()
                            neg_ps_dict[_vs1[0]] = _tmp_set

                        for _vs2 in vs1:

                            if _vs1 != _vs2:
                                _tmp_set = neg_ps_dict[_vs1[0]]
                                _tmp_set.add(_vs2[0])
                                neg_ps_dict[_vs1] = _tmp_set

        ct.print_t("questions_len_train=%s,error_relation_num=%s" % (questions_len_train, error_relation_num))
        self.q_neg_r_tuple = []
        self.q_pos_r_tuple = []
        # questions_len_train = len(self.question_list)
        # counter = ct.generate_counter()
        if is_record and self.mode == "cc":
            p1 = config.cc_par('q_neg_r_tuple')
            with open(p1, mode='w', encoding='utf-8') as o1:
                o1.write('')
        if config.use_property():
            questions_len_train = min(questions_len_train, len(self.entity1_list))
        else:
            questions_len_train = min(questions_len_train, len(self.question_list))
        for index in range(questions_len_train):
            if index % 1000 == 0:
                ct.print("%d / %d" % (index / 1000, questions_len_train), 'build_all_q_r_tuple')
            name = self.entity1_list[index]
            question = self.question_list[index]
            ps_to_except1 = self.relation_path_clear_str_all[index]
            # 待排除的属性可能需要加多
            if self.mode == "wq":
                r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            elif self.mode == "sq":
                r_all_neg = ct.read_entity_and_get_all_neg_relations_sq(entity_id=name, ps_to_except=ps_to_except1)
            elif self.mode == "cc":

                r_all_neg, a_all_neg = self.bh.read_entity_and_get_all_neg_relations_cc(entity_id=name,
                                                                                        ps_to_except=ps_to_except1)

                if build_version == 2:
                    _tmp_set = set()
                    for _p in ps_to_except1:
                        _tmp_set = _tmp_set | set(neg_ps_dict.get(_p, set()))

                    r_all_neg = list((set(r_all_neg) | _tmp_set | neg_ps_set) - set(ps_to_except1))
            else:
                raise Exception("mode error")

            tmp_error_relation_num = min(len(r_all_neg), error_relation_num)
            r_all_neg = r_all_neg[0:tmp_error_relation_num]
            if len(r_all_neg) == 0:
                ct.print("index =%d name:%s " % (index, name), 'r_all_neg')
                # ct.just_log("%s/q_neg_r_tuple_0_error_r.txt" % config.par('sq_fb_path')
                #             , "%s\t%s" % (name, index))

            # print(len(r_all_neg))
            index2 = -1
            for neg_r in r_all_neg:
                index2 += 1
                a1 = a_all_neg[index2]
                q_r_tuple = (index, question, neg_r)
                self.q_neg_r_tuple.append(q_r_tuple)
                if is_record:
                    if self.mode == "wq":
                        ct.just_log("../data/web_questions/q_neg_r_tuple.txt", "%s\t%s" % (question, neg_r))
                    if self.mode == "sq":
                        ct.just_log("%s/q_neg_r_tuple1.txt" % config.par('sq_fb_path')
                                    , "%s\t%s" % (question, neg_r))
                    if self.mode == "cc":
                        ct.just_log(config.cc_par('q_neg_r_tuple')
                                    , "%s\t%s\t%s\t%s" % (index, question, neg_r, a1))
        ct.print_t("build_all_q_r_tuple q_neg_r_tuple")

        # for index in range(questions_len_train):
        #     # name = self.entity1_list[index]
        #     question = self.question_list[index]
        #     ps_to_except1 = self.relation_path_clear_str_all[index]
        #     # r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
        #     for neg_r in ps_to_except1:
        #         q_r_tuple = (question, neg_r)
        #         self.q_pos_r_tuple.append(q_r_tuple)
        #         if is_record:
        #             if self.mode == "wq":
        #                 ct.just_log("../data/web_questions/q_pos_r_tuple.txt", "%s\t%s" % (question, neg_r))
        #             if self.mode == "sq":
        #                 ct.just_log("%s/q_pos_r_tuple.txt" % config.get_sq_topic_path(),
        #                             "%s\t%s" % (question, neg_r))
        # ct.print("build_all_q_r_tuple q_pos_r_tuple")

    # 生成一个epoches中的一个batch;加载全部
    def load_all_q_r_tuple(self, questions_len_train, error_relation_num=9999, is_record=False):
        # 组合所有的问题和错误关系放进一个tuple中
        fname = ""
        if self.mode == "wq":
            fname = "../data/web_questions/q_neg_r_tuple.txt"
        if self.mode == "sq":
            fname = "%s/q_neg_r_tuple.txt" % config.par('sq_fb_rdf_path')
        if self.mode == "cc":
            fname = config.cc_par('q_neg_r_tuple')
        if self.mode == "ner":
            fname = config.cc_par('q_neg_r_tuple')
        # 加载fname
        # 处理
        # 得到self.q_neg_r_tuple
        # q_r_tuple = (index, question, neg_r)
        # self.q_neg_r_tuple.append(q_r_tuple)
        text_lines = ct.file_read_all_lines(fname)
        index = -1
        self.q_neg_r_tuple = []
        for l in text_lines:
            index = int(str(l).split("\t")[0])
            question = str(l).split("\t")[1]
            neg_r = str(l).split("\t")[2].replace("\n", "")
            answer = str(l).split("\t")[3].replace("\n", "")
            q_r_tuple = (index, question, neg_r, answer)
            # if index >questions_len_train:
            #     break # 只载入questions_len_train的问题
            self.q_neg_r_tuple.append(q_r_tuple)

        # self.q_neg_r_tuple = []
        # self.q_pos_r_tuple = []
        # # questions_len_train = len(self.question_list)
        # questions_len_train = min(questions_len_train, len(self.entity1_list))
        #
        # for index in range(questions_len_train):
        #     name = self.entity1_list[index]
        #     question = self.question_list[index]
        #     ps_to_except1 = self.relation_path_clear_str_all[index]
        #     if self.mode == "wq":
        #         r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
        #     elif self.mode == "sq":
        #         r_all_neg = ct.read_entity_and_get_all_neg_relations_sq(entity_id=name, ps_to_except=ps_to_except1)
        #     else:
        #         raise Exception("mode error")
        #     error_relation_num = min(len(r_all_neg), error_relation_num)
        #     r_all_neg = r_all_neg[0:error_relation_num]
        #     for neg_r in r_all_neg:
        #         q_r_tuple = (index, question, neg_r)
        #         self.q_neg_r_tuple.append(q_r_tuple)
        #         if is_record:
        #             if self.mode == "wq":
        #                 ct.just_log("../data/web_questions/q_neg_r_tuple.txt", "%s\t%s" % (question, neg_r))
        #             if self.mode == "sq":
        #                 ct.just_log("%s/q_neg_r_tuple.txt" % config.get_sq_topic_path()
        #                             , "%s\t%s" % (question, neg_r))
        ct.print("load_all_q_r_tuple finish total:%d " % (len(self.q_neg_r_tuple)))

        # for index in range(questions_len_train):
        #     # name = self.entity1_list[index]
        #     question = self.question_list[index]
        #     ps_to_except1 = self.relation_path_clear_str_all[index]
        #     # r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
        #     for neg_r in ps_to_except1:
        #         q_r_tuple = (question, neg_r)
        #         self.q_pos_r_tuple.append(q_r_tuple)
        #         if is_record:
        #             if self.mode == "wq":
        #                 ct.just_log("../data/web_questions/q_pos_r_tuple.txt", "%s\t%s" % (question, neg_r))
        #             if self.mode == "sq":
        #                 ct.just_log("%s/q_pos_r_tuple.txt" % config.get_sq_topic_path(),
        #                             "%s\t%s" % (question, neg_r))
        # ct.print("build_all_q_r_tuple q_pos_r_tuple")

    def build_train_test_q(self):
        for q in self.train_question_list_index:
            q1 = self.converter.arr_to_text_no_unk(q)
            ct.just_log("../data/web_questions/train_questions.txt", q1)

        for q in self.test_question_list_index:
            q1 = self.converter.arr_to_text_by_space(q)
            ct.just_log("../data/web_questions/test_questions.txt", q1)

    # ---
    def test_cap_nums(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b, c = self.cap_nums(a, 0.8)
        ct.print(b)
        ct.print(c)

    def clear_relation(self):
        ct.print("-->clear_relation")
        file_name = "../data/freebase/freebase_relation.txt"
        ct.print(file_name)
        lines = set()
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                line = line.strip()
                if line not in lines:
                    lines.add(line)
                else:
                    ct.print("exist")
        f1_writer = codecs.open("../data/freebase/freebase_relation_clear.txt", mode="w", encoding="utf-8")
        for l in lines:
            f1_writer.write(l + "\n")
        f1_writer.close()

    def log_error_r(self, train, type):
        ct.just_log2("debug", "%s------" % type)
        for _ in train:
            text = self.converter.arr_to_text_by_space(_)
            ct.just_log2("debug", text)

    def log_all_entitys_and_filter_kb(self):
        self.bh = baike_helper()
        self.bh.init_spo(f_in=config.par('cc_kb_path_full'))
        for e in self.entity1_list:
            ct.just_log(config.par('cc_path') + 'e1.txt', e)
            s1 = self.bh.kbqa.get(e, "")
            if s1 == "":
                continue
            for po in s1:
                msg = "%s\t%s\t%s" % (e, po[0], po[1])
                ct.just_log(config.par('cc_path') + 'kb_rdf_only_e.txt', msg)
                # 重写

    # ============== GAN


    # 在用
    def batch_iter_gan_train(self, question_list_index, relation_list_index, model, index,
                             train_part='relation', total=100, pool_mode='additional'):
        """
        20180316 V1.0
        获取NEG的部分改成从全局里面随机获取指定个数
        20180327 V2.0
        获取NEG的部分改成优先从neg的同义词
        :param question_list_index:
        :param relation_list_index:
        :param shuffle_indices:
        :param batch_size:
        :param train_part:
        :return:
        """
        # ct.print("enter:batch_iter_gan_train")

        x_new = []  # 问题集合
        y_pos = []  # 正确属性
        y_neg = []  # 错误属性

        if model == "valid" or model == "train":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
            # index=global_index
        else:
            raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_wq_test_one_debug=================================start")
        try:
            msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        except Exception as e2:
            print(e2)
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        if global_index >= len(self.entity1_list):
            print('error ')
            raise Exception('error')
        name = self.entity1_list[global_index]


        # ps_to_except1 = self.relation_list[global_index]  # 应该从另一个关系集合获取
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的
        # ps_to_except1 = [ps_to_except1]
        padding_num = self.converter.vocab_size - 1
        if pool_mode == 'only_default':
            rs, a_s = self.bh.read_entity_and_get_all_neg_relations_cc(entity_id=name, ps_to_except=ps_to_except1)
        elif pool_mode == 'synonym_train_mode':
            r_pos1 = self.relation_list[global_index]
            rs, a_s = self.bh.rs_gan_synonym(name, ps_to_except1,
                                             total, r_pos1, self.synonym_dict)
        elif pool_mode == 'competing_ps':
            r_pos1 = self.relation_list[global_index]
            rs, a_s = self.bh.competing_ps(r_pos1, ps_to_except1,
                                           total, self.competing_dict,"G")
            if rs is None:
                return None, None, None, None
        else:  # 默认是additional
            rs, a_s = self.bh.rs_cc_gan(entity_id=name, ps_to_except=ps_to_except1,
                                        total=total)

        if pool_mode == 'fixed_amount':
            rs, a_s = rs[0:total], a_s[0:total]
        # ct.print("rs len: %s" % (len(rs)))
        r_len = self.bh.read_entity_and_get_all_neg_relations_cc_len(name, ps_to_except1)

        ct.just_log2("info", "entity:%s " % name)

        r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        q1_text = self.converter.arr_to_text_no_unk(self.question_list_index[global_index])
        r1_msg = "r-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入所有的

        if train_part == 'relation':
            rs = rs
        else:
            rs = a_s
        rs_len = len(rs)
        num = min(config.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        _index = -1
        for r1 in rs:
            _index += 1
            r1_text = r1
            r1_split = [r1]
            r1 = self.converter.text_to_arr_list(r1_split)
            r1_text = self.converter.arr_to_text_no_unk(r1)
            # ct.log3(r1_text)
            # ct.just_log2("info", "r1_neg in test %s" % r1_text)
            # ct.print(r1_text)
            # ct.just_log2("info","neg-r test:" + r1_text)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(self.question_list_index[global_index])
            y_pos.append(self.relation_list_index[global_index])
            y_neg.append(r1)
            # 记录

            #
            # y_new.append(r1)  # neg
            # labels.append(False)
            if pool_mode in ['synonym_train_mode', 'competing_ps']:
                r1_msg = "r-neg: %s" % (r1_text)
            else:
                r1_msg = "r-neg: %s \t answer:%s" % (r1_text, a_s[_index])
            ct.just_log2("info", r1_msg)
            # ct.just_log2("info", ":%s"%self.converter.arr_to_text_no_unk(r1))


        # ct.print("show shuffle_indices")
        ct.just_log2("info","??? len: " + str(len(x_new)) + "  " + str(len(y_pos))+" "+str(len(np.array(y_neg))))
        r_len = len(x_new)
        if len(x_new) == 0:
            print("bug")

        # ct.print("leave:batch_iter_gan_train")
        return np.array(x_new), np.array(y_pos), np.array(y_neg), r_len



    # 20180906-1 生成待训练的CAND_S
    def batch_iter_cand_s(self,model, index,total=100):

        x_new = []  # 问题集合
        y_pos = []  # 正确属性
        y_neg = []  # 错误属性

        # if model == "valid" or model == "train":
        global_index = index
        if global_index >= len(self.entity1_list):
            raise Exception('error')

        # elif model == "test":
        #     global_index = index + self.padding
        # else:
        #     raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_cand_s=================================start")
        msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)


        entity1 = self.entity1_list[global_index]
        cand_s = self.entity1_in_q_cand_list[global_index]
        rs = self.bh.rs_cc_subject(cand_s,total)
        q_current = self.question_list_origin[global_index]

        # 正确的
        s1_in_q = self.entity1_in_q_list[global_index]
        q1_text = q_current        # 问题
        r1_msg = "r-pos: %s \t answer:%s" % (s1_in_q, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", "s1_in_q:%s\tentity1:%s" % (s1_in_q,entity1))
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入所有的

        # if train_part == 'relation':
        #     rs = rs
        # else:
        #     rs = a_s
        rs_len = len(rs)
        num = min(config.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        _index = -1
        for r1 in rs:
            _index += 1

            # 使用原始的问句来避免跟验证属性的冲突
            x_new.append(self.convert_str_to_indexlist(self.question_list_origin[global_index]))
            # 问题 question_list_index[global_index]
            # y_pos.append(self.relation_list_index[global_index])
            y_pos.append(self.convert_str_to_indexlist(s1_in_q))  # 正确的实体
            y_neg.append(self.convert_str_to_indexlist(r1))  # 候选的实体
            # 记录
            #
            # y_new.append(r1)  # neg
            # labels.append(False)
            # if pool_mode in ['synonym_train_mode', 'competing_ps']:

            # else:
            #     r1_msg = "r-neg: %s \t answer:%s" % (r1_text, a_s[_index])
            r1_msg = "r-neg: %s" % (r1)
            ct.just_log2("info", r1_msg)
            # ct.just_log2("info", ":%s"%self.converter.arr_to_text_no_unk(r1))
        # ct.print("show shuffle_indices")
        ct.just_log2("info","len: " + str(len(x_new)) + "  " + str(len(y_pos))+" "+str(len(np.array(y_neg))))
        r_len = len(x_new)
        # if len(x_new) == 0:
        #     print("bug")
        # ct.print("leave:batch_iter_gan_train")
        return np.array(x_new), np.array(y_pos), np.array(y_neg), r_len

    # 20180906-1 生成待训练的CAND_S对应的P
    def batch_iter_cand_s_p(self,model, index,total=100):

        q_ = []  # 问题集合  q_
        q_p = []  # 用于训练属性的问题集合 q_p
        q_s = []  # 用于训练的实体问题集合 q_s
        q_a = []  # 用于答案属性的问题集合
        s_pos = []  # 正确的实体 s_pos
        s_neg = []  # 错误的实体 s_neg
        p_pos = []  # 正确的属性
        p_neg = []  # 错误的属性
        a_pos = []  # 正确的答案
        a_neg = []  # 错误的答案

        # if model == "valid" or model == "train":
        global_index = index
        if global_index >= len(self.entity1_list):
            raise Exception('error')

        # elif model == "test":
        #     global_index = index + self.padding
        # else:
        #     raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_cand_s_p=================================start")
        msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        entity1 = self.entity1_list[global_index]  # 正确的实体(别名词典中的)-字符串
        s1_in_q = self.entity1_in_q_list[global_index]  # 正确的实体(问句中的)-字符串
        cand_s = self.entity1_in_q_cand_list[global_index]  # 候选/错误的实体-list
        # cand_s_neg = self.bh.rs_cc_subject(cand_s, total)  # 错误的实体-字87
        all_cands = [s1_in_q]
        all_cands .extend( cand_s.copy()[0:config.cc_par('ner_top_cand')]) # 前3占了 99%
        # all_cands.append(s1_in_q)  # 全体的候选实体
        q_current = self.question_list_origin[global_index]  # 原始的问题(未处理)-字符串
        r_pos1 = self.relation_list[global_index]  # 正确的属性
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 错误的属性
        a_in_q_pos = self.answer_list[global_index]
        # 增加选出候选的neg属性
        # 得出正确实体的错误属性 和 错误实体的全部属性
        # cand_ps_neg = []
        # cand_ps_neg.extend(ps_to_except1)
        # for cand_s_neg_item in cand_s_neg:
        #     # 得到所有可能的错误属性
        #     temp = self.bh.read_entity_and_get_all_neg_relations_cc(cand_s_neg_item)
        #     cand_ps_neg.extend(temp)
        # cand_ps_neg = list(set(cand_ps_neg))

        r1_msg = "e/r-pos: %s %s \t answer:%s" % (s1_in_q,r_pos1, a_in_q_pos)
        q1_msg = "q : %s" % q_current
        ct.just_log2("info", "s1_in_q:%s\tentity1:%s" % (s1_in_q,entity1))
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)
        # 加入所有的

        # if train_part == 'relation':
        #     rs = rs
        # else:
        #     rs = a_s
        # rs_len = len(cand_s_neg)
        # num = min(config.get_static_num_debug(), rs_len)
        # cand_s_neg = cand_s_neg[0:num]

        c_temp_cand_ps_neg, c_temp_cand_ps_neg = [],[]
        if config.cc_par('negative_sampling_model') == 'competing':
            ct.print('\nr_pos1:%s' % r_pos1, 'choice')
            neg_size = 10
            c_temp_cand_ps_neg, c_temp_cand_as_neg \
                = self.bh.competing_ps(r_pos1, [r_pos1],
                                       neg_size, self.competing_train_dict, "G")
            for _1, _2 in \
                    zip(c_temp_cand_ps_neg, c_temp_cand_ps_neg):
                _1 = _1[0]
                _2 = _2[0]
                ct.print(_1,'choice')
        _index = 0
        for cand_s_neg_item in all_cands:
            # 增加选出候选的neg属性
            # 得出正确实体的错误属性 和 错误实体的全部属性
            # key 是指存在于字典中的KEY
            if config.cc_par('use_alias_dict'):
                temp_cand_ps_neg,temp_cand_as_neg = \
                    self.bh.kb_get_p_o_by_s(cand_s_neg_item)
            else:
            # 如果使用别名字典 则不需要替换回来
                if s1_in_q == cand_s_neg_item :
                    d_key = entity1
                else:
                    d_key = cand_s_neg_item
                temp_cand_ps_neg, temp_cand_as_neg = \
                        self.bh.read_entity_and_get_all_neg_relations_cc(d_key)
                raise Exception('目前只使用别名词典的办法获取属性')
                # self.bh.read_entity_and_get_all_neg_relations_cc(d_key)
            # 前提是 s 是正确的
            if cand_s_neg_item == s1_in_q :
                for _1, _2 in \
                        zip(c_temp_cand_ps_neg, c_temp_cand_ps_neg):
                    _1 = _1[0]
                    _2 = _2[0]
                    ct.print(_1,'choice')
                    # 考虑在这里去除重复
                    if _1 not in temp_cand_ps_neg:
                        temp_cand_ps_neg.append(_1)
                        temp_cand_as_neg.append(_2)
                    else:
                        ct.print('%s 重复出现'% (_1), 'temp')
            for _cand_ps_neg_item,_as in \
                    zip(temp_cand_ps_neg,temp_cand_as_neg):
                # 如果实体和属性都是正确的，则跳过
                if cand_s_neg_item == s1_in_q and _cand_ps_neg_item == r_pos1:
                    # ct.print("check: %s - %s"%(cand_s_neg_item,temp_cand_ps_neg_item))
                    continue
                if _as == a_pos:
                    continue
                _index += 1
                # 使用原始的问句来避免跟验证属性的冲突
                q_.append(self.convert_str_to_indexlist(q_current))  # 待增加模式替换对应S
                q_s.append(self.convert_str_to_indexlist(q_current.replace(_cand_ps_neg_item, '♢')))
                q_current_for_p = q_current.replace(cand_s_neg_item, '♠')  # 去掉实体的问句,用于属性训练
                q_p.append(self.convert_str_to_indexlist(q_current_for_p))
                q_current_for_a = q_current_for_p.replace(_cand_ps_neg_item, '♢')  # 去掉属性的问句,用于属性训练
                q_a.append(self.convert_str_to_indexlist(q_current_for_a))
                # q_current_for_e = str(q_current).replace('♠', s1_in_q)  # 去掉属性的问句,用于实体训练
                # 问题 question_list_index[global_index]
                # y_pos.append(self.relation_list_index[global_index])
                s_pos.append(self.convert_str_to_indexlist(s1_in_q))  # 正确的实体
                s_neg.append(self.convert_str_to_indexlist(cand_s_neg_item))  # 候选的实体
                p_pos.append(self.convert_str_to_indexlist(r_pos1))
                p_neg.append(self.convert_str_to_indexlist(_cand_ps_neg_item))
                a_pos.append(self.convert_str_to_indexlist(a_in_q_pos))
                a_neg.append(self.convert_str_to_indexlist(_as))

                r1_msg = "s-neg,r-neg,a-neg: %s - %s - %s \t q_current_for_a:%s " % \
                         (cand_s_neg_item,_cand_ps_neg_item,_as,q_current_for_a)
                ct.just_log2("info", r1_msg)
                need_return = _index % total == 0 and _index != 0
                if need_return:
                    d1 = self.return_dict(a_neg, a_pos, p_neg, p_pos, q_, q_a, q_p, q_s, s_neg, s_pos)
                    q_.clear()
                    q_p.clear()
                    q_s.clear()
                    q_a.clear()
                    s_pos.clear()
                    s_neg.clear()
                    p_pos.clear()
                    p_neg.clear()
                    a_pos.clear()
                    a_neg.clear()
                    yield d1
            # end for
            # if len(q_)>0:
            #     d1 = self.return_dict(a_neg, a_pos, p_neg, p_pos, q_, q_a, q_p, q_s, s_neg, s_pos)
            #     yield d1

    def return_dict(self, a_neg, a_pos,  p_neg, p_pos, q_, q_a, q_p, q_s, s_neg, s_pos):
        d1 = dict()
        d1['q_'] = np.array(q_)  # 0
        d1['q_p'] = np.array(q_p)  # 1
        d1['q_s'] = np.array(q_s)  # 2
        d1['q_a'] = np.array(q_a)  # 2
        d1['s_pos'] = np.array(s_pos)  # 3
        d1['s_neg'] = np.array(s_neg)  # 4
        d1['p_pos'] = np.array(p_pos)  # 5
        d1['p_neg'] = np.array(p_neg)  # 6
        d1['a_pos'] = np.array(a_pos)  # 5
        d1['a_neg'] = np.array(a_neg)  # 6

        return d1

    # 1
    # 在用，valid_batch_debug-> 生成一个问题的相关信息
    # 专门用于 entity_relation，同时生成候选实体和候选属性
    # 去掉 yield 改成 一次返回完全
    def batch_iter_cc_ner_entitiy_test_one(self,  model, index,total=100):
        q_ = []  # 问题集合
        q_p = []  # 用于训练属性的问题集合（剔除实体）
        q_s = []  # 用于训练实体的问题集合（剔除属性）
        q_a = []
        s_pos = []  # 正确的实体
        s_neg = []  # 错误的实体
        p_pos = []  # 正确的属性
        p_neg = []  # 错误的属性
        a_pos = []  # 正确的答案
        a_neg = []  # 错误的答案

        if model == "valid":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
        else:
            raise Exception("MODEL 参数出错")
        if global_index >= len(self.entity1_list):
            raise Exception('error')

        # elif model == "test":
        #     global_index = index + self.padding
        # else:
        #     raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_cc_ner_entitiy_test_one=================================start")
        msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        entity1 = self.entity1_list[global_index]  # 正确的实体(别名词典中的)-字符串
        s1_in_q = self.entity1_in_q_list[global_index]  # 正确的实体(问句中的)-字符串
        # 测试的时候应该遵循真实顺序，而非强行加入
        # cand_s = self.entity1_in_q_cand_list[global_index]  # 正确的实体-字符串
        # cand_s_neg = self.bh.rs_cc_subject(cand_s, total)  # 错误的实体-字符串
        all_cands = self.entity1_in_q_cand_list_origin[global_index]  # 正确的实体-字符串
        all_cands = all_cands[0:3]  # 我们使用前3个来做训练
        # all_cands = [s1_in_q] #
        # all_cands .extend( cand_s.copy()[0:config.cc_par('ner_top_cand')]  ) # 前3占了 99%

        # all_cands.append(s1_in_q)  # 全体的候选实体
        q_current = self.question_list_origin[global_index]  # 原始的问题(未处理)-字符串
        r_pos1 = self.relation_list[global_index]  # 正确的属性
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 错误的属性
        labels = []  # 标签
        ner_labels = []  # ner 标签
        rel_labels = []
        a_in_q_pos = self.answer_list[global_index]

        # 增加选出候选的neg属性
        # 得出正确实体的错误属性 和 错误实体的全部属性
        # cand_ps_neg = []
        # cand_ps_neg.extend(ps_to_except1)
        # for cand_s_neg_item in cand_s_neg:
        #     # 得到所有可能的错误属性
        #     temp = self.bh.read_entity_and_get_all_neg_relations_cc(cand_s_neg_item)
        #     cand_ps_neg.extend(temp)
        # cand_ps_neg = list(set(cand_ps_neg))

        r1_msg = "r-pos: %s \t answer:%s" % (r_pos1, a_in_q_pos)
        q1_msg = "q : %s" % q_current
        ct.just_log2("info", "s1_in_q:%s\tentity1:%s" % (s1_in_q,entity1))
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)


        _index = 0
        for cand_s_neg_item in all_cands:
            if config.cc_par('use_alias_dict'):
                temp_cand_ps_neg,temp_cand_as_neg = \
                    self.bh.kb_get_p_o_by_s(cand_s_neg_item)
            else:
                # 如果使用别名字典 则不需要替换回来
                if s1_in_q == cand_s_neg_item :
                    d_key = entity1
                else:
                    d_key = cand_s_neg_item
                temp_cand_ps_neg, temp_cand_as_neg = \
                        self.bh.read_entity_and_get_all_neg_relations_cc(d_key)

            for _cand_ps_neg_item,_as in \
                    zip(temp_cand_ps_neg,temp_cand_as_neg):

                _index += 1
                # 使用原始的问句来避免跟验证属性的冲突
                q_.append(self.convert_str_to_indexlist(q_current))  # 待增加模式替换对应S
                q_current_for_p = q_current.replace(cand_s_neg_item,'♠')  # 去掉实体的问句,用于属性训练
                q_p.append(self.convert_str_to_indexlist(q_current_for_p))
                q_s.append(self.convert_str_to_indexlist(q_current.replace(_cand_ps_neg_item, '♢')))
                q_current_for_a = q_current_for_p.replace(_cand_ps_neg_item, '♢')  # 去掉属性的问句,用于属性训练
                q_a.append(self.convert_str_to_indexlist(q_current_for_a))

                # q_current_for_e = str(q_current).replace('♠', s1_in_q)  # 去掉属性的问句,用于实体训练
                # 问题 question_list_index[global_index]
                # y_pos.append(self.relation_list_index[global_index])
                s_pos.append(self.convert_str_to_indexlist(s1_in_q))  # 正确的实体
                s_neg.append(self.convert_str_to_indexlist(cand_s_neg_item))  # 候选的实体
                # 增加选出候选的neg属性
                # 得出正确实体的错误属性 和 错误实体的全部属性
                p_pos.append(self.convert_str_to_indexlist(r_pos1))
                p_neg.append(self.convert_str_to_indexlist(_cand_ps_neg_item))
                a_pos.append(self.convert_str_to_indexlist(a_in_q_pos))
                a_neg.append(self.convert_str_to_indexlist(_as))
                # 如果实体和属性都是正确的，则跳过
                if cand_s_neg_item == s1_in_q and _cand_ps_neg_item == r_pos1:
                        # and _as == a_in_q_pos:
                    # ct.print("check: %s - %s"%(cand_s_neg_item,temp_cand_ps_neg_item))
                    # continue
                    labels.append(True)
                else:
                     labels.append(False)
                if cand_s_neg_item == s1_in_q :
                    ner_labels.append(True)
                else:
                    ner_labels.append(False)
                if _cand_ps_neg_item == r_pos1:
                    rel_labels.append(True)
                else:
                    rel_labels.append(False)
                r1_msg = "s-neg,r-neg,a-neg: %s - %s - %s \t q_current_for_a:%s " % \
                         (cand_s_neg_item,_cand_ps_neg_item,_as,q_current_for_a)
                ct.just_log2("info", r1_msg)
        # end for
        d1 = dict()
        d1['q_'] = np.array(q_)   # 0
        d1['q_p'] = np.array(q_p)   # 1 # 用于训练属性的问题集合（无实体）
        d1['q_s'] = np.array(q_s)   # 2
        d1['q_a'] = np.array(q_a)  #
        d1['s_pos'] = np.array(s_pos)   # 3
        d1['s_neg'] = np.array(s_neg)   # 4
        d1['p_pos'] = np.array(p_pos)   # 5
        d1['p_neg'] = np.array(p_neg)   # 6
        d1['labels'] = labels  # 7
        d1['ner_labels'] = ner_labels  # 7
        d1['rel_labels'] = rel_labels  # 7
        d1['a_pos'] = np.array(a_pos)  # 5
        d1['a_neg'] = np.array(a_neg)  # 6

        # assert labels.__contains__(True) # 必须包含一个正确的答案?
        if not labels.__contains__(True):
            ct.print('%s\t%s\t%s '%(q_current,r_pos1,a_in_q_pos),'bad_q')
        return d1

    # 获取竞争属性集合，用于计算互相之间的相似度
    def batch_iter_competing_ps_cosine(self):

        total = 1000
        _index = 0
        r_pos = []
        r_cp = []
        r_cp_str = ''
        prob2 = []

        for p_k, p_v in self.competing_train_dict.items():
            ps_to_except = [p_k]
            r_cp_str = p_k
            for _p in p_v: # ( 属性 ，个数)
                if _p[0] not in ps_to_except:
                    _index += 1
                    r_pos.append(self.convert_str_to_indexlist(p_k))  # 加自己
                    r_cp.append(self.convert_str_to_indexlist(_p[0]))
                    prob2.append(_p[1])
                need_return = _index % total == 0
                if need_return:
                    d1 = dict()
                    d1['r_pos'] = np.array(r_pos)
                    d1['r_cp'] = np.array(r_cp)

                    d1['r_pos_str'] = r_cp_str
                    r_pos.clear()
                    r_cp.clear()
                    yield d1
        if len(r_pos) > 0:
            d1 = dict()
            d1['r_pos_str'] = r_cp_str
            d1['r_pos'] = np.array(r_pos)
            d1['r_cp'] = np.array(r_cp)
            yield d1

    # 对照 feature2cos_sim 写成 np 版本
    def cal_cosine(self,p1,p2):
        # cosine=x*y/(|x||y|)
        norm_p1 = np.sqrt(np.sum(np.multiply(p1,p1)))
        norm_p2 = np.sqrt(np.sum(np.multiply(p2, p2)))
        mul_p1_p2 = np.sum(np.multiply(p1,p2))
        cos_sim_p1_p2 = np.divide(mul_p1_p2,np.multiply(norm_p1,norm_p2))
        return cos_sim_p1_p2

    # 更新和计算竞争属性集的每个属性的top k
    # d1  key = 属性，value = 属性对应的向量
    # 同时更新一份到 competing_train_dict 供对接之前的前TOP K个概率出
    def update_competing_ps_cosine(self,top_n,d1):
        cp_dict = dict()  # key = 属性 value =  top-k 的属性
        competing_train_dict_new = dict()
        for p_k, p_v in self.competing_train_dict.items():
            ps_to_except = [p_k]
            p_k_xl = d1.get(p_k,'')
            if p_k_xl == '':
                continue
            st_list = []
            _index = -1
            _competing_train_dict_set = set()
            for _p in p_v:  # ( 属性 ，个数)
                # _p = p_v[i]
                if _p[0] in ps_to_except:
                    continue
                _index += 1
                _p_xl = d1.get(_p[0],'')
                if _p_xl == '':
                    continue
                st = ct.new_struct()
                st.index = _index
                st.p = _p[0]
                st.score = np.mean(self.cal_cosine(p_k_xl,_p_xl))
                st_list.append(st)
                # _competing_train_dict 部分
                _tp =(st.p,st.score)
                _competing_train_dict_set.add(_tp)

            # 排序
            st_list.sort(key=ct.get_key)
            st_list.reverse()
            if len(st_list)>top_n:
                st_list = st_list[0:top_n]
            cp_dict[p_k] = st_list
            ct.print("@%s" % p_k, 'update_score')
            for item in st_list:
                ct.print("%s\t%s"%(item.p,str(item.score)),'update_score')
            competing_train_dict_new[p_k] = _competing_train_dict_set

        self.cp_dict = cp_dict
        self.competing_train_dict = competing_train_dict_new



    # 获取竞争属性集合，用于计算互相之间的相似度
    def batch_iter_all_competing_ps(self):
        total = 1000
        _index = 0
        r_cp = []
        for item in self.competing_set:
            _index += 1
            r_cp.append(self.convert_str_to_indexlist(item))
            need_return = _index % total == 0
            if need_return:
                d1 = dict()
                d1['r_cp'] = np.array(r_cp)
                r_cp.clear()
                yield d1

        if len(r_cp) > 0:
            d1 = dict()
            d1['r_cp'] = np.array(r_cp)
            yield d1


    # 更新competing ps
    def update_train_competing_ps_cosine(self,r_pos,r_cp,top_n,st_list):
        # d1 = dict()   # key = pos_p , value = top k
        st_list = st_list[0:top_n]
        _s = set()
        ct.print(str(r_pos), 'update_train_competing_ps_cosine')
        for st in st_list :
            _item = (r_cp[st.index],st.score)
            _s.add(_item)
            ct.print(str(_item),'update_train_competing_ps_cosine')
        self.competing_train_score_dict[r_pos] = _s

    # 获取一个SP 的所有答案
    # TP  FP FN
    def get_o_from_kb(self,model,index):
        if model == "valid":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
        else:
            raise  Exception('NO')

        entity1 = self.entity1_list[global_index]
        s1_in_q = self.entity1_in_q_list[global_index]  # 正确的实体(问句中的)-字符串
        r_pos1 = self.relation_list[global_index]  # 正确的属性
        # cand_s = self.entity1_in_q_cand_list[global_index]
        # rs = self.bh.rs_cc_subject(cand_s,total)
        # q_current = self.question_list_origin[global_index]
        a_in_q_pos = self.answer_list[global_index]

        s = entity1
        p = r_pos1
        o = a_in_q_pos

        tp, fp = \
            self.bh.kb_get_spo_by_s_p(s1_in_q,s,p,o)
        # fn = (s,p,o)
        fn = "%s\t%s\t%s"%(s,p, o)
        return  tp,fp,fn

    # 同义词模块
    def init_synonym(self, f1='../data/nlpcc2016/5-class/demo1/same_p_tj.v3.txt',
                     f2='../data/nlpcc2016/5-class/demo1/same_p_tj_clear_dict.txt'):
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []
        synonym_dict = dict()
        synonym_score_dict = dict()
        for x in f1s:
            try:
                k1 = x.split('\t')[0]
                k2 = x.split('\t')[1]
            except Exception as e1:
                print(e1)
            synonym_dict = ct.dict_add(synonym_dict, k1, k2)
            synonym_dict = ct.dict_add(synonym_dict, k2, k1)

            synonym_score_dict["%s\t%s" % (k1, k2)] = x.split('\t')[4]
        # for k in synonym_dict.keys():
        #     msg = "%s\t%s" % (k, '\t'.join(synonym_dict[k]))
        #     f2s.append(msg)
        # ct.file_wirte_list(f2, f2s)
        self.synonym_dict = synonym_dict
        self.synonym_score_dict = synonym_score_dict

    def synonym_train_data(self, f1):
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []
        synonym_train_dict = dict()
        synonym_train_keys = []
        synonym_train = []
        for x in f1s:
            k1 = x.split('\t')[0]
            v1 = x.split('\t')[1:]
            # synonym_train_dict[k1] = v1
            synonym_train_keys.append(k1)
            synonym_train.append(v1)

        self.synonym_train_keys = synonym_train_keys
        self.synonym_train = synonym_train
        # self.synonym_train_dict = synonym_train_dict
        ct.print('synonym_train_dict ok')

    def convert_str_to_indexlist(self, r1):
        padding_num = self.converter.vocab_size - 1
        r1_split = [r1]  # .split(" ")
        r1 = self.converter.text_to_arr_list(r1_split)
        # r1_text = self.converter.arr_to_text_no_unk(r1)
        # ct.log3(r1_text)
        # ct.just_log2("info", "r1_neg in test %s" % r1_text)
        # ct.print(r1_text)
        # ct.just_log2("info","neg-r test:" + r1_text)
        r1 = ct.padding_line(r1, self.max_document_length, padding_num)
        return r1

    def convert_str_to_indexlist_2(self, r1, padding=True):
        padding_num = self.converter.vocab_size - 1
        # r1_split = r1.split(" ")
        r1_split = [x for x in r1]
        r1 = self.converter.text_to_arr_list(r1_split)
        # r1_text = self.converter.arr_to_text_no_unk(r1)
        # ct.log3(r1_text)
        # ct.just_log2("info", "r1_neg in test %s" % r1_text)
        # ct.print(r1_text)
        # ct.just_log2("info","neg-r test:" + r1_text)
        if padding:
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
        return r1

    # 产生s model下的数据
    def batch_iter_s_model(self, index):
        train_q = []
        train_cand = []
        train_neg = []
        td = self.synonym_train[index]
        max_num = 10

        if len(td) > max_num:
            td_index = np.random.choice(len(self.synonym_train[index]), size=max_num, replace=False)
            td_new = []
            for _i in td_index:
                td_new.append(td[_i])
            td = td_new
        num = len(td)
        neg_data = ct.synonym_random_get(self.synonym_train_keys, self.synonym_train, td, num)
        if len(neg_data) < len(td):
            print('error !!')

        for i in range(len(td)):
            q1 = self.convert_str_to_indexlist(self.synonym_train_keys[index])
            train_q.append(q1)  # key
            v1 = self.convert_str_to_indexlist(td[i])
            train_cand.append(v1)
            neg1 = self.convert_str_to_indexlist(neg_data[i])
            train_neg.append(neg1)
            msg = "%d\t%s\t%s\t%s" % (index, self.synonym_train_keys[index], td[i], neg_data[i])
            ct.print(msg, 'debug_batch_iter_s_model')
        ct.print('----', 'debug_batch_iter_s_model')
        return np.array(train_q), np.array(train_cand), np.array(train_neg)

    # competing model 竞争模块
    def init_competing_model(self, f1='competing_ps_path'):
        competing_dict = dict()
        # competing_prob_dict = dict()
        competing_set = set()
        f1s = ct.file_read_all_lines_strip_no_tips(f1)
        for l1 in f1s:
            k1 = str(l1).split('\t')[0]  # key
            v1 = str(l1).split('\t')[1]  #
            competing_set.add(v1)
            _prob = float(str(l1).split('\t')[2])
            if competing_dict.__contains__(k1):
                _cs_set=competing_dict[k1]
            else:
                _cs_set = set()
            _cs_set.add((v1,_prob))
            competing_dict[k1] = _cs_set
            # competing_prob_dict["%s_%s"%(k1,v1)] = _prob
        self.competing_train_dict = competing_dict
        self.competing_set = competing_set

    # competing model 竞争模块
    # 错误方法
    def init_competing_model_bak(self, f1='competing_ps_path'):
        competing_dict = dict()
        f1s = ct.file_read_all_lines_strip(f1)
        for l1 in f1s:
            k1 = str(l1).split('\t')[0]
            v1 = str(l1).split('\t')[1:]
            competing_dict[k1] = set(v1)
        self.competing_dict = competing_dict
    # def init_alias_dict(self,f2=''):
    #     # # 构造字典
    #     alias_dict = dict()
    #     f2s = ct.file_read_all_lines_strip(f2)  # 字典
    #     for l2 in f2s:
    #         _s = str(l2).split('\t')[0]
    #         _ps = str(l2).split('\t')[1:]
    #         alias_dict[_s] = _ps
    #     self.alias_dict = alias_dict

    # # 20180906-1 用cos做NER
    # def init_cos_ner_model(self, f1='ner_path'):
    #     self.cand_s = []
    #     f1s = ct.file_read_all_lines_strip(f1)
    #     for l1 in f1s:
    #         self.cand_s.append(str(l1).split('\t'))

    # 产生c model下的数据
    def batch_iter_competing_ps(self, model, index,
                                train_part='relation', total=100, pool_mode='additional'):

        # ct.print("enter:batch_iter_gan_train")

        x_new = []  # 问题集合
        y_pos = []  # 正确属性
        y_neg = []  # 错误属性

        if model == "valid" or model == "train":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
            # index=global_index
        else:
            raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_competing_ps=================================start")
        try:
            msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        except Exception as e2:
            print(e2)
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        if global_index >= len(self.entity1_list):
            print('error ')
            raise Exception('error')
        name = self.entity1_list[global_index]


        # ps_to_except1 = self.relation_list[global_index]  # 应该从另一个关系集合获取
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的
        # ps_to_except1 = [ps_to_except1]
        padding_num = self.converter.vocab_size - 1

        r_pos1 = self.relation_list[global_index]
        rs, a_s = self.bh.competing_ps(r_pos1, ps_to_except1,
                                       total, self.competing_dict,"C")
        if len(rs) == 0:
            ct.print('%s not exist ' % r_pos1)
            return None

        # if pool_mode == 'fixed_amount':
        #     rs, a_s = rs[0:total], a_s[0:total]
        # ct.print("rs len: %s" % (len(rs)))
        # r_len = self.bh.read_entity_and_get_all_neg_relations_cc_len(name, ps_to_except1)

        ct.just_log2("info", "entity:%s " % name)

        r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        q1_text = self.converter.arr_to_text_no_unk(self.question_list_index[global_index])
        r1_msg = "r-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入所有的

        if train_part == 'relation':
            rs = rs
        else:
            rs = a_s
        rs_len = len(rs)
        num = min(config.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        _index = -1
        for r1 in rs:
            _index += 1
            r1_text = r1
            r1_split = [r1]
            r1 = self.converter.text_to_arr_list(r1_split)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(self.question_list_index[global_index])
            y_pos.append(self.relation_list_index[global_index])
            y_neg.append(r1)
            r1_msg = "r-neg: %s" % (r1_text)
            ct.just_log2("info", r1_msg)
            if _index % total == 0 and _index != 0:
                x_new_ret = x_new.copy()
                y_pos_ret = y_pos.copy()
                y_neg_ret = y_neg.copy()

                x_new.clear()
                y_pos.clear()
                y_neg.clear()

                yield np.array(x_new_ret), np.array(y_pos_ret), np.array(y_neg_ret)

                # ct.print("show shuffle_indices")
                # ct.print("len: " + str(len(x_new)) + "  " + str(len(y_pos)))
                # ct.print("leave:batch_iter_gan_train")
                # return np.array(x_new), np.array(y_pos), np.array(y_neg), r_len

    # 产生a model下的数据
    def batch_iter_additional_ps(self, model, index,
                                train_part='relation', total=100, pool_mode='additional'):

        # ct.print("enter:batch_iter_gan_train")

        x_new = []  # 问题集合
        y_pos = []  # 正确属性
        y_neg = []  # 错误属性

        if model == "valid" or model == "train":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
            # index=global_index
        else:
            raise Exception("MODEL 参数出错")

        # log
        ct.just_log2("info", "\nbatch_iter_additional_ps=================================start")
        try:
            msg = "model=%s\tid=%s\tglobal_index=%d\tq_global_index=%d" % (
                model, index, global_index, self.question_global_index[global_index])
        except Exception as e2:
            print(e2)
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        if global_index >= len(self.entity1_list):
            print('error ')
            raise Exception('error')
        name = self.entity1_list[global_index]


        # ps_to_except1 = self.relation_list[global_index]  # 应该从另一个关系集合获取
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的
        # ps_to_except1 = [ps_to_except1]
        padding_num = self.converter.vocab_size - 1

        r_pos1 = self.relation_list[global_index]
        rs, a_s = self.bh.rs_cc_gan(r_pos1, ps_to_except1,total) # 重点
        if len(rs) == 0:
            ct.print('%s not exist ' % r_pos1)
            return None

        # if pool_mode == 'fixed_amount':
        #     rs, a_s = rs[0:total], a_s[0:total]
        # ct.print("rs len: %s" % (len(rs)))
        # r_len = self.bh.read_entity_and_get_all_neg_relations_cc_len(name, ps_to_except1)

        ct.just_log2("info", "entity:%s " % name)

        r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        q1_text = self.converter.arr_to_text_no_unk(self.question_list_index[global_index])
        r1_msg = "r-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入所有的

        if train_part == 'relation':
            rs = rs
        else:
            rs = a_s
        rs_len = len(rs)
        num = min(config.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        _index = -1
        for r1 in rs:
            _index += 1
            r1_text = r1
            r1_split = [r1]
            r1 = self.converter.text_to_arr_list(r1_split)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(self.question_list_index[global_index])
            y_pos.append(self.relation_list_index[global_index])
            y_neg.append(r1)
            r1_msg = "r-neg: %s" % (r1_text)
            ct.just_log2("info", r1_msg)
            if _index % total == 0 and _index != 0:
                x_new_ret = x_new.copy()
                y_pos_ret = y_pos.copy()
                y_neg_ret = y_neg.copy()

                x_new.clear()
                y_pos.clear()
                y_neg.clear()

                yield np.array(x_new_ret), np.array(y_pos_ret), np.array(y_neg_ret)

                # ct.print("show shuffle_indices")
                # ct.print("len: " + str(len(x_new)) + "  " + str(len(y_pos)))
                # ct.print("leave:batch_iter_gan_train")
                # return np.array(x_new), np.array(y_pos), np.array(y_neg), r_len

    # char-rnn
    def build_vocab_ner(self):
        # 建造词汇表
        # 将问题和关系的字符串变成以空格隔开的一个单词的list
        # total_list = self.question_list + self.relation_list
        # q_words = self.get_split_list(self.question_list)
        q_words = []

        for q in self.question_list_origin:
            # q = str(q).replace("\n\r", " ")
            q_words_list = [x for x in q]
            for word in q_words_list:
                q_words.append(word)

        q_words.extend(['ἐ', 'ল', '♠'])
        # q_words.extend(self.get_split_list(self.relations))  # freebase里面的关系
        # 应该再加上问题里面的关系集合
        # q_words = [str(x).replace(".","") for x in q_words ]
        self.converter = read_utils.TextConverter(q_words)

    # 将x转为id x
    def convert_x_to_x_new(self, x):
        x_new = []
        for _ in x:
            _new = []
            for _1 in _:
                _new1 = []
                # for _2 in _1:
                _3 = self.convert_str_to_indexlist_2(_1)
                _new.append(_3)
                # _new.append(_new1)
            x_new.append(_new)

        return x_new

    def batch_iter_char_rnn(self, total):
        # 只取前面的训练
        qs_train = self.question_list[0:config.cc_par('real_split_train_test_skip')]
        # arr = np.array(qs.copy())
        # batch_size = n_seqs * n_steps
        # n_batches = int(len(arr) / batch_size)
        # arr = arr[:batch_size * n_batches]
        # arr = arr.reshape((n_seqs, -1))
        if len(qs_train) == 0:
            raise ('qs_train = 0')

        x = []
        y = []
        index = 0
        for q in qs_train:
            index += 1
            # ct.print('epoches:%d ' % epoches)
            _tmp_x = "ἐ%sল" % q  # 使用2个无意义的字符做开始和结束的字符

            _tmp_y = [x for x in _tmp_x]
            _tmp_x = [x for x in _tmp_x]
            del _tmp_y[0]

            x.append(self.convert_str_to_indexlist_2(''.join(_tmp_x)))
            y.append(self.convert_str_to_indexlist_2(''.join(_tmp_y)))

            # 将问题转化
            if index % total == 0:
                r_x = x.copy()
                r_y = y.copy()
                x.clear()
                y.clear()
                # print(len(r_x))
                yield np.array(r_x), np.array(r_y)



                # for n in range(0, arr.shape[1], n_steps):
                #     x = arr[:, n:n + n_steps]
                #     y = np.zeros_like(x)
                #     y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
                #     # 转为
                #     x_new = self.convert_x_to_x_new(x)
                #     y_new = self.convert_x_to_x_new(y)
                #     yield x_new, y_new

    # 答案选择模块
    def init_expend_es(self, f1='../data/nlpcc2016/4-ner/extract_entitys_all_tj.resort_3.expend.v1.txt'):
        f1s = ct.file_read_all_lines_strip(f1)
        # f1s_new = []
        # # f1s = [_ for _ in filter(lambda x: str(x).__contains__('NULL'), f1s)]

        expend_es = []
        expend_score = []
        s1 = '____'
        for l1 in f1s:
            vs = str(l1).split('\t')
            index = -1
            # d1 = dict()
            list1 = []
            list2 = []
            for vs1 in vs:
                index += 1
                # d1[str(index)] = vs1[1:] # KEY=第几个，value 对应的实体
                vs2 = str(vs1).split(s1)
                #  截取分数之后的
                vs2 = vs2[2:]
                list1.append(vs2)
                list2.append(vs2[1])
            expend_score.append(list2)
            expend_es.append(list1)
        self.expend_es = expend_es
        self.expend_score = expend_score
        ct.print('init_expend_es len:%d' % len(expend_es))


# ======================================================================= clear data


def test_batch_iter():
    # d = DataClass("wq")
    d = DataClass("wq")
    # id, ps = d.find_entity_and_relations_paths(path=r"D:\ZAIZHI\freebase-data\topic-json", entity_id="012bg5")
    # ct.print(id)
    # ct.print(ps)

    # d.batch_iter(d.train_question_list_index, d.train_relation_list_index,
    #              batch_size=10)
    for i in range(1):
        # a = d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index, batch_size=10)
        q = []
        q.append([1, 2, 3])
        q.append([1, 2, 3])
        q.append([1, 2, 3])
        r1 = q.copy()
        r2 = q.copy()
        a = d.batch_iter_wq_debug_fix_model(q, r1, r2, 1)
        for aa in a:
            # ct.print(aa[0])
            # ct.print(aa[1])
            # ct.print(aa[2])
            ct.print(1)
            # d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index, "valid")

            #
            # d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
            #                       batch_size=10)
            # d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index,
            #                                batch_size=10)




            # ct.print(z)

            # d.compare()
            # d = DataClass("debug")
            # e1 = d.find_entity("100_classic_book_collection"+".json.gz")
            # ct.print(e1)
            # ct.print(d.batch_iter(2))


def test_random_choose_indexs_debug():
    d = DataClass("cc")
    for i in range(20):
        my_generator = d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
                                             batch_size=10)
        for gen in my_generator:
            print(gen)
            break
        d.batch_iter_wq_test_one_debug(d.test_question_list_index, d.test_relation_list_index, "test", 1)
        break


def test_log_e():
    dh = DataClass("cc")
    dh.log_all_entitys_and_filter_kb()


def test_build():
    d = DataClass("wq")
    filename1 = '../data/word2vec/train.model.1516630487.7132027'
    filename2 = '../data/word2vec/wiki.vector'

    # d.prodeuce_embedding_vec_file(filename1) 生成wiki.vector文件
    # a1, a2, a3 = d.build_embedding_weight(filename2)
    # d.build_all_q_r_tuple() 只需要运行一次
    # d.build_train_test_q()
    ct.print("111")


def test_sq():
    dh = DataClass("sq")
    dh.build_all_q_r_tuple(config.get_static_q_num_debug(),
                           config.get_static_num_debug(), is_record=True)
    # my_generator = dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index,10)
    # for gen in my_generator:
    #     train_q = gen[0]
    #     train_cand = gen[1]
    #     train_neg = gen[2]
    #     # print(train_q)
    #     print("ok")
    print(0000000000)


# 初始化
def test_cc():
    dh = DataClass("cc")
    shuffle_indices = np.random.permutation(np.arange(len(dh.q_neg_r_tuple_train)))  # 打乱样本下标
    my_generator = dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index, shuffle_indices,
                                          batch_size=10)
    for gen in my_generator:
        train_q = gen[0]
        train_cand = gen[1]
        train_neg = gen[2]
        print("%s\t%s\t%s" % (train_q, train_cand, train_neg))
        break

    batchsize = 10
    evaluate_batchsize = 10
    i = 0
    for model in ['valid', 'test']:
        i += 1
        if model == "valid":
            id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
            question_list_index = dh.train_question_list_index
            relation_list_index = dh.train_relation_list_index

        else:
            id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))
            question_list_index = dh.test_question_list_index
            relation_list_index = dh.test_relation_list_index

        id_list = ct.random_get_some_from_list(id_list, evaluate_batchsize)
        print(id_list)
        batchsize = min(batchsize, len(id_list))
        for i in range(batchsize):
            index = id_list[i]

            dh.batch_iter_wq_test_one_debug(question_list_index,
                                            relation_list_index,
                                            model, index)


def init_cc():
    dh = DataClass(mode="cc", run_type='init')
    # 只需要构建一次
    dh.build_all_q_r_tuple(99999999999999,
                           99999999999999, is_record=True)


def test_gan():
    dh = DataClass("cc")
    train_part = config.cc_par('train_part')
    model = 'train'
    step = 0
    train_step = 0
    batch_size = 100
    shuffle_indices = np.random.permutation(np.arange(len(dh.train_question_list_index)))  # 打乱样本下标
    shuffle_indices = [x for x in list(shuffle_indices)]
    # 1 遍历raw
    for index in shuffle_indices:
        # 取出一个问题的相关数据
        train_q, train_pos, train_neg, r_len = dh.batch_iter_gan_train(dh.train_question_list_index,
                                                                       dh.train_relation_list_index, model,
                                                                       index, train_part, batch_size)
        question = ''
        relations = []
        for _ in train_q:
            v_s_1 = dh.converter.arr_to_text_no_unk(_)
            valid_msg = model + " test_q 1:" + v_s_1
            ct.print(valid_msg, "debug")
            question = v_s_1
            break
        for _ in train_pos:
            v_s_1 = dh.converter.arr_to_text_no_unk(_)
            valid_msg = model + " pos 1:" + v_s_1
            ct.print(valid_msg, "debug")
            relations.append(v_s_1)
            break
        for _ in train_neg:
            v_s_1 = dh.converter.arr_to_text_no_unk(_)
            valid_msg = model + " neg 1:" + v_s_1
            ct.print(valid_msg, "debug")
            relations.append(v_s_1)
        break


def test_tyc():
    dh = DataClass("cc")
    dh.init_synonym()


def test_ner():
    dh = DataClass("ner")
    g = dh.batch_iter_char_rnn(1)
    for x, y in g:
        print(x)
        print(y)


def test_answer():
    dh = DataClass("cc")
    train_question_list_index = None
    train_relation_list_index = None
    model = 'valid'
    index = 1
    train_part = 'relation'
    g = dh.batch_iter_cc_answer_test_one_debug(
        train_question_list_index, train_relation_list_index, model, index,
        train_part)
    for x, y, z in g:
        print(x)
        print(y)
        print(z)

def test_ner_entitiy():
    dh = DataClass("cc")
    r_pos1 = '性别'
    rs, a_s = dh.bh.competing_ps(r_pos1, [r_pos1],
                                   10, dh.competing_train_dict, "G")

    for _cand_ps_neg_item, _as in \
            zip(rs, a_s):
        _cand_ps_neg_item=_cand_ps_neg_item[0]
        print(_cand_ps_neg_item)
    # print(rs)
    train_question_list_index = None
    train_relation_list_index = None
    model = 'valid'
    index = 1
    train_part = 'entity'
    g = dh.batch_iter_cand_s(model, index,10)
    for x, y, z,m in g:
        break
        print(x)
        print(y)
        print(z)
    pass

if __name__ == "__main__":
    # CC 部分的测试-和构建代码
    #    init_cc()

    # test_random_choose_indexs_debug()
    # test_random_choose_indexs_debug()
    # test_cc()
    # test_answer()
    test_ner_entitiy()
    # 测试生成

    # test_random_choose_indexs_debug()

    # 记录所有的实体
    # test_log_e()


    # a = read_rdf_from_gzip_or_alias(path=r"F:\3_Server\freebase-data\topic-json", file_name="1")
    # ct.print(a)
    # clear_relation()
    # test2()
    # test_random_choose_indexs_debug()
    # clear_relation()
