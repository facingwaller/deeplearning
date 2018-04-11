from lib.ct import ct
from lib.config import config
import os
import gzip
import gc
import math
import numpy as np


# 1 装载数据

class lr_helper:
    def extract_line(self, f1_l):

        f1_ls = str(f1_l).split('\t')
        split_word = '____'
        # 0____1____8.718868____9.006550____机械设计基础____8
        step = ''  # f1_ls[0]
        model = ''  # f1_ls[1]
        global_index = 0  # f1_ls[2]
        question = f1_ls[0]
        ds = f1_ls[8:]  # str(f1_l).split('\t')  #
        ts = []
        index = -1
        for item in ds:
            index += 1
            try:
                right = str(item).split(split_word)[1] == '1'
            except Exception as e1:
                print(e1)
            if right:
                right1 = [1, 0]
            else:
                right1 = [0, 1]
            # 0____1____8.718868____9.006550____机械设计基础____8
            score1 = float(str(item).split(split_word)[2])
            score2 = str(item).split(split_word)[3]
            relation = str(item).split(split_word)[4]
            # z_score = ct.get_zi_flag_score_ps(question, relation)
            score3 = str(item).split(split_word)[5]  # ct.math2(question, relation)

            t1 = (index, right1, relation, right, score1, score2, score3)
            ts.append(t1)
        t1 = (global_index, question, ts, step, model)
        return t1

    def __init__(self, f1='',
                 f2='../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'):
        self.all_datas = []
        # f1 = '../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'
        f1s = ct.file_read_all_lines_strip(f1)
        self.train_data = []
        self.test_data = []
        index = -1
        for f1_l in f1s:
            index += 1
            need_skip = False
            if str(f1_l).__contains__('NULL'):
                need_skip = True
            if str(f1_l).__contains__('####'):
                need_skip = True
            if str(f1_l).__contains__('@@@@@@'):
                need_skip = True

            if need_skip:
                continue

            if index < config.cc_par('real_split_train_test_skip'):  # <= int(len(f1s)*0.8):
                # if str(f1_l).__contains__('\tvalid\t'):
                self.train_data.append(self.extract_line(f1_l))
            else:
                self.test_data.append(self.extract_line(f1_l))

        print('init ok')

    def baseline(self, data):
        total = len(data)
        # shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标

        right_index = 0
        rith_answer1 = 0
        rith_answer2 = 0
        rith_answer3 = 0
        rith_answer4 = 0
        for list_index in range(total):
            index = -1
            data_current = data[list_index]
            # for ts in data_current[2]:
            #     index += 1
            # 问题  Z分数 NN得分
            if int(data_current[2][0][1][0]) == 1:
                rith_answer1 += 1
            elif int(data_current[2][1][1][0]) == 1:
                rith_answer2 += 1
            elif int(data_current[2][2][1][0]) == 1:
                rith_answer3 += 1
            else:
                rith_answer4 += 1
                # else:
                #     print(data_current)

        return [rith_answer1 / total, (rith_answer1 + rith_answer2) / total,
                (rith_answer1 + rith_answer2 + rith_answer3) / total,
                (rith_answer1 + rith_answer2 + rith_answer3 + rith_answer4) / total]

    def batch_iter(self, data, batch_size):
        total = len(data)
        shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标

        info1 = "q total:%d ; epohches-size:%s " % (total, len(data) // batch_size)
        ct.print(info1, 'info')
        x_new = []
        y_new = []
        z_new = []
        p_new = []

        rith_answer = 0
        right_index = 0
        for list_index in range(total):
            index = -1
            data_current = data[shuffle_indices[list_index]]
            for ts in data_current[2]:
                index += 1
                x_new.append(data_current[1])
                # 0____1____8.718868____9.006550____机械设计基础____8
                # t1 = (index, right1, score, relation,right,z_score)
                y_new.append((float(ts[4]), float(ts[5]), float(ts[6])))  # 继续遍历
                z_new.append(ts[1])
                p_new.append(ts[4])
                # ts

                # 问题  Z分数 NN得分
                if int(ts[1][0]) == 1:
                    right_index = index
                    rith_answer += 1
                msg = "%s\t%s\t%s\t%s\t%s" % (data_current[1], ts[5], ts[2], ts[3], ts[4])
                ct.print(msg, 'debug1')

            # if list_index % batch_size == 0 and list_index != 0:
            x_return = x_new.copy()  # 问题
            y_return = y_new.copy()  # 数据
            z_return = z_new.copy()  # 标签
            p_return = p_new.copy()  # 属性

            x_new.clear()
            y_new.clear()
            z_new.clear()
            p_new.clear()
            yield np.array(x_return), np.array(y_return), np.array(z_return), np.array(p_return), right_index


if __name__ == "__main__":
    batch_size = 1
    lh = lr_helper(f1='../data/nlpcc2016/4-ner/demo3/q.rdf.ms.re.top_99.v4.10.txt')

    print(lh.baseline(lh.train_data))
    print(lh.baseline(lh.test_data))

    gc1 = lh.batch_iter(lh.train_data, batch_size)
    #
    # for gc2 in gc1:
    #     x1 = gc2[1]
    #     y1 = gc2[2]
    #     x1 = x1.reshape(-1, 2)
    #     y1 = y1.reshape(-1, 2)
    #     print(x1)
    #     print(y1)
