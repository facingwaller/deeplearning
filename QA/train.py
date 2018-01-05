# QA的NN
# author:ender
# 第一版，简单的LSTM做QA问答关联

import tensorflow as tf
import lib.data_helper as data_helper
import QA.custom_nn as mynn
import os
import codecs

# 定义变量
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("batch_size", "", 'path')
tf.flags.DEFINE_string('input_file_train', '../data/simple_questions/annotated_fb_data_train-1.txt', 'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_test', '', 'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_freebase', '', 'utf8 encoded text file')


# 主流程
def main():
    # 1 读取所有的数据,返回一批数据标记好的数据{data.x,data.label}
    # batch_size 是1个bath，questions的个数，
    dh = data_helper.DataClass()

    # all_data = dh.load_all_train_data() # 加载所有训练数据
    # bath_x = dh.embedding(bath_x)  # embedding
    # bath = dh.next_bath() #获取一个批次的数据
    # dh.load_test_data() # 加载测试数据

    # 3 构造模型LSTM类
    # 4 设定loss
    # 5 执行
    print(1)


if __name__ == '__main__':
    main()
