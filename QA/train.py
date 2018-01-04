# QA的NN
# author:ender
# 第一版，简单的LSTM做QA问答关联

import tensorflow as tf
import lib.data_helper as data_helper
import os
import codecs

# 定义变量
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("path", "", 'path')


# 主流程
def main():
    # 1 读取数据,返回一批数据标记好的数据{data.x,data.label}
    # batch_size 是1个bath，questions的个数，
    # data = data_helper.train_batch_iter(batch_size)
    # 2 embedding
    # 3 构造模型LSTM类
    # 4 设定loss
    # 5 执行
    print(1)


if __name__ == '__main__':
    main()
