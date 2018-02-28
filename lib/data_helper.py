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

        self.rdf_list = []
        self.mode = mode
        if mode == "test":
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train.txt")
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_test.txt")
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_valid.txt")
            self.init_fb("../data/freebase/")
        elif mode == "small":
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train-small.txt")
            # self.init_fb("../data/freebase/")
        elif mode == "wq":
            self.init_web_questions()
            ct.print("init_web_questions finish.")
            self.init_fb("../data/freebase/")
            ct.print("init_fb finish.")
            # 初始化排除的关系
            # self.init_filter_relations()
        elif mode == "sq":
            self.init_simple_questions(config.par('sq_q_path'))
            ct.print("init_simple_questions finish.")
            self.init_fb(config.par('sq_fb_path'))
            ct.print("init_fb finish.")
        elif mode == "cc":

            need_load_kb = True
            if need_load_kb:
                self.bh = baike_helper()
                self.bh.init_spo(f_in=config.cc_par('kb-use'))

            self.init_cc_questions(config.cc_par('cc_q_path'), run_type)
            ct.print("init_cc_questions finish.")
            self.converter = read_utils.TextConverter(filename=config.par('cc_vocab'), type="zh-cn")
            if run_type == 'init':  # 初始化
                return
            msg = 'questions_len_train:%s\twrong_relation_num:%s\t' % (
                config.get_static_q_num_debug(), config.get_static_num_debug())
            ct.print(msg, 'debug')

            self.load_all_q_r_tuple(config.get_static_q_num_debug(), config.get_static_num_debug(), is_record=True)
            self.get_max_length()
            self.q_r_2_arrary_and_padding()
            # 按比例分割训练和测试集
            self.division_data(0.8, config.cc_par('real_split_train_test'))
            self.build_embedding_weight(config.wiki_vector_path(mode))
            #
            ct.print("load embedding ok!")

            return
        else:
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train-1.txt")
            self.init_fb("../data/freebase/fb_1000/")

        # 建造词汇表
        self.build_vocab()
        # 获取最大长度
        self.get_max_length()
        # 问题和关系都转换成array形式，并padding问题
        self.q_r_2_arrary_and_padding()
        # 按比例分割训练和测试集
        self.division_data()

        self.build_embedding_weight(config.wiki_vector_path(mode))
        ct.print("load embedding ok!")

        self.build_all_q_r_tuple(config.get_static_q_num_debug(),
                                 config.get_static_num_debug(), is_record=False)

        # self.load_all_q_r_tuple(config.get_static_q_num_debug(),
        #                         config.get_static_num_debug(), is_record=False)

        # ct.print("build_all_q_r_tuple 生成所有的q和neg r的组合")

    def q_r_2_arrary_and_padding(self):
        # 把问题和关系变成array形式
        self.question_list_split = self.get_split_list_per_line(self.question_list)
        self.relation_list_split = self.get_split_list_per_line(self.relation_list)
        for q_l_s in self.question_list_split:
            self.question_list_index.append(self.converter.text_to_arr_list(q_l_s))
        # self.relation_list_index = self.converter.text_to_arr(self.relation_list_split)
        for _ in self.relation_list_split:
            self.relation_list_index.append(self.converter.text_to_arr_list(_))
        # 第一版本先padding到max长度
        padding_num = self.get_padding_num()
        for index in range(0, len(self.question_list_index)):
            self.question_list_index[index] = \
                ct.padding_line(self.question_list_index[index], self.max_document_length, padding_num)
        for index in range(0, len(self.relation_list_index)):
            self.relation_list_index[index] = \
                ct.padding_line(self.relation_list_index[index], self.max_document_length, padding_num)
            # for s in self.relation_list_index:
            #     s = ct.padding_line(s,self.max_document_length,padding_num)
            # 截断

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
        print('看看哪些neg关系是训练有')
        r_train = set()
        r_test = set()
        ct.print('train', 'train_test_q')
        ct.print('\t%d\t%s\t%s\t%s\t%s' % (0, '实体', '关系', '问题', '被替换词'), 'train_test_q')
        for l in range(len(self.train_question_list_index)):
            ct.print("\t%d\t%d\t%s\t%s\t%s\t%s" % (self.question_global_index[l],l, self.entity1_list[l], self.relation_list[l],
                                               self.question_list[l], self.entity_ner_list[l]),
                     'train_test_q')
            r_train.add(self.relation_list[l])
        ct.print('test', 'train_test_q')
        for l in range(len(self.test_question_list_index)):
            global_index = l + self.padding
            ct.print("\t%d\t%d\t%s\t%s\t%s\t%s" % (self.question_global_index[global_index],
                global_index, self.entity1_list[global_index], self.relation_list[global_index],
                self.question_list[global_index], self.entity_ner_list[global_index]), 'train_test_q')
            r_test.add(self.relation_list[global_index])
        ct.print('test not in train', 'train_test_q')
        # 看看哪些pos关系是训练有，测试没有的
        r3 = (r_train | r_test) - r_train
        for r in r3:
            ct.print(r, 'train_test_q')

        # 看看哪些neg关系是训练有，测试没有的
        ct.print('neg test not in train', 'train_test_q')
        neg_r_train = set()
        neg_r_test = set()
        for l in range(len(self.train_question_list_index)):
            global_index = l
            ps_to_except1 = self.relation_path_clear_str_all[global_index]
            rs = self.bh.read_entity_and_get_all_neg_relations_cc(self.entity1_list[global_index], ps_to_except1)
            for r in rs:
                neg_r_train.add(r)

        for l in range(len(self.test_question_list_index)):
            global_index = l + self.padding
            ps_to_except1 = self.relation_path_clear_str_all[global_index]
            rs = self.bh.read_entity_and_get_all_neg_relations_cc(self.entity1_list[global_index], ps_to_except1)
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
        else:
            # 将问题/关系转换成index的系列表示
            max_document_length1 = max([len(x.split(" ")) for x in self.question_list])  # 获取单行的最大的长度
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
        idx = 0
        # 《机械设计基础》这本书的作者是谁？    杨可桢，程光蕴，李仲生
        # 机械设计基础         作者          杨可桢，程光蕴，李仲生
        # 问题0 答案1 实体s-2 关系p-3 属性值o-4    匹配到的实体s-5
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file.readlines():
                    idx += 1
                    line_seg = line.split('\t')
                    if len(line_seg) < 6 or line.__contains__('NULL'):  # todo:rewrite input file,重写输入文件
                        ct.print("bad:" + line, "bad")
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
            f3s = ct.file_read_all_lines_strip(config.cc_par('rdf_extract_property_test'))
            # 选出所有的实体，筛选一遍f2s
            f3s = [str(x).split('\t')[0] for x in f3s]
            f2s_new = []
            for x in f2s:
                if str(x).split('\t')[0] in f3s:
                    f2s_new.append(x)
            f2s = f2s_new
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
                test_id = int(str(f4s[i]).split('\t')[1])
                ct.print('test_id = %d'%test_id, 'train_test_q')
                f3s = str(f4s[i]).split('\t')[2:]
                break

            ct.just_log(config.cc_par('rdf_maybe_property_index'), str(index+1))
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
            line_seg = line.split('\t')
            answer = line_seg[1]
            entity1 = line_seg[2]
            relation1 = ct.clean_str_rel(line_seg[3].lower())  # 清洗关系
            entity2 = line_seg[4]
            entity_ner = line_seg[5].replace('\n', '').replace('\r', '')
            # todo: if char can't convert ,filter them,如果需要转换不了，到时候在这里直接过滤
            # 6.1.1.3 3 在载入问题的时候用♠替换掉实体
            question = line_seg[0]
            question = question.replace(' ', '').lower()
            if not question.__contains__(entity_ner):
                ct.print(question, 'entity_ner')
            question = question.replace(entity_ner, '♠')

            self.entity1_list.append(entity1)
            self.relation_list.append(relation1)
            self.entity2_list.append(entity2)

            self.question_list.append(question)  # 将问题替换掉
            self.entity_ner_list.append(entity_ner)
            self.answer_list.append(answer)

            # todo:111
            # 针对CC的排除关系 ，需要遍历找出其他的属性
            # if entity1 == '对酒':
            #     print(543535)
            rs1 = [relation1]
            vs = self.bh.kbqa.get(entity1, '')
            if vs != '':
                for k, v in vs:
                    if v == answer:
                        if k not in rs1:
                            rs1.append(k)
            self.relation_path_clear_str_all.append(rs1)
            # self.rdf_list.append([entity1, relation1, entity2])
            # check it
            # line_list.append(line)

            # 增加一个容器 标记所有的问题是否属于训练集合还是测试集合
            is_train = index > config.cc_par('real_split_train_test_skip')
            self.question_labels.append(is_train)

            self.question_global_index.append(index)

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

    # --------------------生成batch
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
    # todo: train data
    def batch_iter_wq_debug(self, question_list_index, relation_list_index, batch_size=100):
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
        x_new = []
        y_new = []
        z_new = []
        # self.q_neg_r_tuple 这个地方需要筛选出仅有问题列表里面的数据
        # todo : bug here

        # 生成 0- len(question_list_index) 的随机数字
        total = len(self.q_neg_r_tuple_train)
        shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标

        info1 = "q total:%d ; epohches-size:%s " % (total, len(self.q_neg_r_tuple_train) / batch_size)
        ct.print(info1, 'info')

        for list_index in range(total):
            # 获取q_neg_r_tuple里面打乱的下标的对应的 q_r 对
            q_neg_r = self.q_neg_r_tuple_train[shuffle_indices[list_index]]
            index = q_neg_r[0]  # 对应类里面的index
            name = q_neg_r[1]  # 问题
            r_neg = q_neg_r[2]  # 关系
            print(index)
            x_new.append(x[index])  # 添加问题
            y_new.append(y[index])  # 添加正确的关系
            ct.print(x[index], "debug_epoches")
            ct.print(y[index], "debug_epoches")

            r1 = self.converter.text_to_arr_list(r_neg)  # 文字转数字
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
            msg_neg = "r-neg %d :%s       " % (list_index, r_neg)
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
            x_new.append(error_test_q_list[shuffle_indices[list_index]])
            y_new.append(error_test_pos_r_list[shuffle_indices[list_index]])
            z_new.append(error_test_neg_r_list[shuffle_indices[list_index]])
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

    # -- 老版本 纯随机
    def batch_iter_wq_debug_random(self, question_list_index, relation_list_index, batch_size=100):
        """
        web questions 的生成反例的办法。debug版本，
        生成指定batch_size的数据。

        update:
        1. 改为1个epoches获取不到重复的数据.--2018年1月23日11:56:22

        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_debug_random")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []

        id_list = ct.get_static_id_list_debug()  # 获取指定的index
        #  这里应该使用随机从里面提取一个 打乱样本
        shuffle_indices = ct.random_get_some_from_list(id_list, batch_size)
        # 下面需要指定每个index和neg关系
        # 取消 # 到这里还是变成1个 也就是一次还是跑1个问题-
        # msg1 = "\n batch_iter_wq_debug index q= %s " % self.shuffle_indices_debug
        # ct.just_log2("info",msg1)
        total = 0
        for index in shuffle_indices:
            # index = self.shuffle_indices_debug  # 将随机到的赋值给当前
            x_new.append(x[index])
            y_new.append(y[index])
            # 对于z_new 加进去
            #   根据index，寻找entity里面非relaiton的relation[]
            # s1 获取entity;这个index是问句的index，问句对应了entity-name
            # （self.entity1_list）
            # s2 根据entity-name直接读取entity-id的gzip
            # 现在有2种方法，1 直接读取；2 或者

            question = self.question_list[index]
            name = self.entity1_list[index]

            # ct.print(name)
            # self.converter.arr_to_text_by_space(self.relation_list_index[2808])
            # ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
            ps_to_except1 = self.relation_path_clear_str_all[index]

            # ps_to_except1 = [ps_to_except1]
            # length = len(x[index])
            r1, r1_index = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
            r1_text = r1
            r1 = self.converter.text_to_arr_list(r1)

            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())

            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            z_new.append(r1)

            info1 = "q=%d ,r-pos=%d,r-neg=%d q=%s e=%s  %d,%d" % (
                index, index, r1_index, question, name, len(ps_to_except1), len(r_all_neg))
            ct.print(info1[0:30], "debug")
            ct.just_log2("info", info1)
            msg = "qid=%d,neg r=%d  " % (index, r1_index)
            ct.log3(msg)
            for r in ps_to_except1:
                # ct.print("r-pos %d :%s       " % (len(str(r).split(" ")), r))
                ct.just_log2("info", "r-pos %d :%s       " % (len(str(r).split(" ")), r))
            # for r in r_all_neg:
            #      ct.just_log2("info","r-neg %d :%s       " % (len(str(r1).split(" ")), r))

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            ct.just_log2("info", msg_neg)
            # ct.print(msg_neg)
            # ct.just_log2("info","=======================================")
            total += 1
            if total >= batch_size:
                break

        ct.print(shuffle_indices[0:batch_size])
        #  ct.just_log2("info","======================================= end train data build")
        ct.print("leave:batch_iter_wq_debug")
        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --第二版，使得每次产生的不重复
    # --------------------生成batch
    # def batch_iter_init(self):
    #     length = len(self.question_list_index)
    #     self.shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
    #     self.shuffle_index = 0  # 索引
    #     ct.print(1)

    # def can_batch_wq(self, batch_size):
    #     if self.shuffle_index + batch_size > len(self.shuffle_indices):
    #         return False
    #     else:
    #         return True

    # --------------------生成test的batch
    # --------------------生成batch 暂时不用
    # def batch_iter_wq_test_one(self, question_list_index, relation_list_index, batch_size=100):
    #     """
    #     web questions
    #     生成指定batch_size的数据
    #     :param batch_size:
    #     :return:
    #     """
    #
    #     x = question_list_index.copy()
    #     y = relation_list_index.copy()
    #     x_new = []  # 问题集合
    #     y_new = []  # 关系集合
    #     z_new = []  #
    #     labels = []  # 标签集合
    #     shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本
    #     # ct.print("shuffle_indices", str(shuffle_indices))
    #
    #     total = 0
    #     index = shuffle_indices[0]  # 选取第一个
    #
    #     msg = "test id=%s " % index
    #     ct.print(msg)
    #     ct.log3(msg)
    #     ct.just_log2("info",msg)
    #
    #     name = self.entity1_list[index]
    #
    #     ps_to_except1 = self.relation_list[index]
    #     ps_to_except1 = [ps_to_except1]
    #     padding_num = self.get_padding_num()  # self.converter.vocab_size - 1
    #     # r1 = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
    #     rs = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
    #
    #     rs = list(set(rs))
    #     # 加入正确的
    #     x_new.append(x[index])
    #     y_new.append(y[index])
    #     labels.append(True)
    #
    #     # log
    #     r1_text = self.converter.arr_to_text_by_space(y[index])
    #     ct.print("r-pos: %s" % r1_text)
    #     ct.just_log2("info","r-pos: %s" % r1_text)
    #
    #     # 加入错误的,暂时加入控制免得太多
    #     if ct.is_debug_few():
    #         num = min(ct.get_static_num_debug(), len(rs))
    #     else:
    #         num = len(rs)
    #     rs = rs[0:num]
    #     for r1 in rs:
    #         r1 = self.converter.text_to_arr_list(r1)
    #         r1_text = self.converter.arr_to_text_by_space(r1)
    #         r1 = ct.padding_line(r1, self.max_document_length, padding_num)
    #         x_new.append(x[index])
    #         y_new.append(r1)  # neg
    #         labels.append(False)
    #
    #         # ct.log3(r1_text)
    #         ct.just_log2("info","r1_neg in test %s" % r1_text)
    #
    #     # ct.print("11111111111111111111111111")
    #     # ct.print(len(r1))
    #     # z_new.append(r1)
    #     #
    #     # total += 1
    #     # if total >= batch_size:
    #     #         break
    #     # ct.print("show shuffle_indices")
    #     # ct.print(shuffle_indices[0:batch_size])
    #     # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
    #     # rate = 1
    #     # r_si = reversed(shuffle_indices)
    #     # r_si = list(r_si)
    #     # ct.print(r_si)
    #     # total = 0
    #     # for index in r_si:
    #     #     z_new.append(y[index])
    #     #     total += 1
    #     #     if total >= batch_size:
    #     #         break
    #     ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
    #
    #     return np.array(x_new), np.array(y_new), np.array(labels)

    # todo: test data
    def batch_iter_wq_test_one_debug(self, question_list_index, relation_list_index, model, index):
        """
        web questions
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_test_one_debug")
        # x = question_list_index.copy()
        # y = relation_list_index.copy()
        x_new = []  # 问题集合
        y_new = []  # 关系集合
        z_new = []  #
        labels = []  # 标签集合

        if model == "valid":
            global_index = index
        elif model == "test":
            global_index = index + self.padding
            # index=global_index
        else:
            raise Exception("MODEL 参数出错")

        # shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        # ct.print("shuffle_indices", str(shuffle_indices))

        # 使用这个问题的index作为测试的问题
        # index = index

        # index = self.shuffle_indices_debug
        # 从debug的index集合里面随机挑选一个
        # id_list = []
        # if model == "valid":
        #     id_list = ct.get_static_id_list_debug()
        # elif model == "test":
        #     id_list = ct.get_static_id_list_debug_test()
        # else:
        #     raise Exception("MODEL 参数出错")
        # ct.print("这里暂时用的外面传进来的index")
        # index = ct.random_get_one_from_list(id_list)
        # index = shuffle_indices[0]
        # 当前给一个
        # x_new.append(x[index])
        # y_new.append(y[index])

        # log
        ct.just_log2("info", "\nbatch_iter_wq_test_one_debug=================================start")
        msg = "model=%s,id=%s,global_index=%d;q_global_index=%d;" % (
            model, index, global_index, self.question_global_index[global_index])
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        # 这个index应该要偏移出训练集
        # if self.converter.
        # arr_to_text_by_space(x[index]).replace(' ','').replace('<unk>','')\
        #         == self.question_list[global_index]:
        #     # 部分语句中有空格的会不相等，
        if global_index >= len(self.entity1_list):
            print(1111)
        name = self.entity1_list[global_index]
        # todo: index should not in
        # ps_to_except1 = self.relation_list[global_index]  # 应该从另一个关系集合获取
        ps_to_except1 = self.relation_path_clear_str_all[global_index]  # 从这里拿是对的
        # ps_to_except1 = [ps_to_except1]
        padding_num = self.converter.vocab_size - 1
        # r1 = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
        if self.mode == "wq":
            rs = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
        if self.mode == "sq":
            rs = ct.read_entity_and_get_all_neg_relations_sq(entity_id=name,
                                                             ps_to_except=ps_to_except1, not_allow_repeat=True)
        if self.mode == "cc":
            rs = self.bh.read_entity_and_get_all_neg_relations_cc(entity_id=name, ps_to_except=ps_to_except1)

        # rs = list(set(rs))
        # 加入正确的
        # if index >= len(x):
        #     print(3132131)
        x_new.append(self.question_list_index[global_index])
        y_new.append(self.relation_list_index[global_index])
        labels.append(True)
        # ct.print("batch_iter_wq_test_one_debug ")

        ct.just_log2("info", "entity:%s " % name)
        # ct.just_log2("info","relation:%s " % name)

        # ct.print(y[index])
        r1_text = self.converter.arr_to_text_no_unk(self.relation_list_index[global_index])
        q1_text = self.converter.arr_to_text_no_unk(self.question_list_index[global_index])
        r1_msg = "r-pos: %s \t answer:%s" % (r1_text, self.answer_list[global_index])
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入错误的
        # todo : total is get_static_num_debug
        rs_len = len(rs)
        num = min(ct.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        for r1 in rs:
            r1_split = r1.split(" ")
            r1 = self.converter.text_to_arr_list(r1_split)
            r1_text = self.converter.arr_to_text_no_unk(r1)
            # ct.log3(r1_text)
            ct.just_log2("info", "r1_neg in test %s" % r1_text)
            # ct.print(r1_text)
            # ct.just_log2("info","neg-r test:" + r1_text)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(self.question_list_index[global_index])
            y_new.append(r1)  # neg
            labels.append(False)

        # ct.print("show shuffle_indices")
        ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
        ct.print("leave:batch_iter_wq_test_one_debug")
        return np.array(x_new), np.array(y_new), np.array(labels)

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
        # 组合所有的问题和错误关系放进一个tuple中
        # self.question_list
        # self.relation_path_clear_str_all 正确关系
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
                # todo: add answer to filter more
                r_all_neg = self.bh.read_entity_and_get_all_neg_relations_cc(entity_id=name, ps_to_except=ps_to_except1)
            else:
                raise Exception("mode error")

            tmp_error_relation_num = min(len(r_all_neg), error_relation_num)
            r_all_neg = r_all_neg[0:tmp_error_relation_num]
            if len(r_all_neg) == 0:
                ct.print("index =%d name:%s " % (index, name), 'r_all_neg')
                # ct.just_log("%s/q_neg_r_tuple_0_error_r.txt" % config.par('sq_fb_path')
                #             , "%s\t%s" % (name, index))

            # print(len(r_all_neg))

            for neg_r in r_all_neg:
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
                                    , "%s\t%s\t%s" % (index, question, neg_r))
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
            q_r_tuple = (index, question, neg_r)
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

    my_generator = dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index,
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


if __name__ == "__main__":
    # CC 部分的测试-和构建代码
    # init_cc()

    # test_random_choose_indexs_debug()
    # test_random_choose_indexs_debug()
    test_cc()

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
