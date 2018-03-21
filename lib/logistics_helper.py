from lib.ct import ct
from lib.config import config
import os
import gzip
import gc
import math
import numpy as np


# 1 装载数据

class logistics_helper:
    def extract_line(self, f1_l):

        f1_ls = str(f1_l).split('\t')
        step = f1_ls[0]
        model = f1_ls[1]
        global_index = f1_ls[2]
        question = f1_ls[3]
        ds = f1_ls[4:]
        ts = []
        for item in ds:
            index = int(str(item).split('_')[0])
            right = str(item).split('_')[1] == '1'
            if right:
                right1 = [1, 0]
            else:
                right1 = [0, 1]
            score = float(str(item).split('_')[2])
            relation = str(item).split('_')[3]

            # z_score = ct.get_zi_flag_score_ps(question, relation)
            z_score = ct.math2(question, relation)

            t1 = (index, right1, score, relation, right, z_score)
            ts.append(t1)
        t1 = (global_index, question, ts, step, model)
        return t1

    def __init__(self, f1='../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt',
                 f2= '../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'):
        self.all_datas = []
        # f1 = '../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'
        f1s = ct.file_read_all_lines_strip(f1)
        self.train_data = []
        self.test_data = []
        for f1_l in f1s:
            if str(f1_l).__contains__('\tvalid\t'):
                self.train_data.append(self.extract_line(f1_l))
            else:
                self.test_data.append(self.extract_line(f1_l))

        print('init ok')

    def baseline(self,data):
        total = len(data)
        # shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标
        rith_answer = 0
        right_index = 0
        rith_answer = 0
        right_index = 0
        for list_index in range(total):
            index = -1
            data_current = data[list_index]
            # for ts in data_current[2]:
            #     index += 1
                # 问题  Z分数 NN得分
            if int(data_current[2][0][1][0]) == 1:
                    rith_answer += 1

        return rith_answer/total
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
                # t1 = (index, right1, score, relation,right,z_score)
                y_new.append((ts[2], ts[5]))  # 继续遍历
                z_new.append(ts[1])
                p_new.append(ts[3])
                # ts

                # 问题  Z分数 NN得分
                if int(ts[1][0])==1:
                    right_index = index
                    rith_answer+=1
                msg = "%s\t%s\t%s\t%s\t%s" % (data_current[1], ts[5], ts[2], ts[3],ts[4])
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
            yield np.array(x_return), np.array(y_return), np.array(z_return),np.array(p_return),right_index




if __name__ == "__main__":
    batch_size = 2
    lh = logistics_helper()

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

