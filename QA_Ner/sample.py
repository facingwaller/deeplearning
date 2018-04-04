from QA_Ner.model import CharRNN
import os
import codecs
import tensorflow as tf
import lib.data_helper as data_helper
import QA.custom_nn as mynn
import time
import datetime
import numpy as np
from lib.ct import ct
# from lib.config import FLAGS
from lib.config import config
import os
from QA_GAN.QACNN import QACNN
from QA_GAN.Discriminator import Discriminator
from QA_GAN.Generator import Generator
from lib.ct import ct

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 100, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 1, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
# tf.flags.DEFINE_integer('embedding_size', 100, 'size of embedding')
# tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '../char_rnn/data/c1_train.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train 100000 这里是epoch')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps 100')
tf.flags.DEFINE_integer('log_every_n', 100, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 99999999, 'max char number')

tf.flags.DEFINE_string('checkpoint_path', 'model/default/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '♠是', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 10, 'max length to generate')


def prepare_data():
    f1 = '../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt'
    f3 = '../data/nlpcc2016/4-ner/extract_entitys_all_tj.txt'
    f4 = '../data/nlpcc2016/4-ner/extract_entitys_all_tj.sort_by_ner_lstm.txt'
    f1s = ct.file_read_all_lines_strip(f1)
    f3s = ct.file_read_all_lines_strip(f3)
    f1s_new = []
    f3s_new = []
    for i in range(len(f1s)):
        if str(f1s[i]).__contains__('NULL'):
            continue
        f1s_new.append(f1s[i])
        f3s_new.append(f3s[i])

    # 过滤NULL
    # 获取候选实体逐个去替代和判断

    # cs.append('立建候时么什是♠')
    # 读取出所有候选实体并打分取出前3 看准确率

    f4s = []
    _index = -1
    for l1 in f1s_new:  # 遍历每个问题
        _index += 1
        replace_qs = []
        for l3 in f3s_new[_index].split('\t'):
            q_1 = str(l1).split('\t')[0].replace(l3, '♠')
            replace_qs.append((q_1, l3))
        entitys = []
        for content, l3 in replace_qs:
            # content = input("input:")
            r1 = '1'
            entitys.append((l3, r1))
            # print(content)
            # print(r1)
            # print(score_list)
        entitys.sort(key=lambda x: x[1])
        entitys_new = [x[0] for x in entitys]

        f4s.append('\t'.join(entitys_new))
    ct.file_wirte_list(f4, f4s)


def main(_):
    # prepare_data()
    # FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    # converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = \
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    model = 'ner'
    dh = data_helper.DataClass(model)
    train_batch_size = 1
    # g = dh.batch_iter_char_rnn(train_batch_size)  # (FLAGS.num_seqs, FLAGS.num_steps)
    embedding_weight = dh.embeddings

    model = CharRNN(dh.converter.vocab_size,  # 词汇表大小 从其中生成所有候选
                    num_seqs=train_batch_size,  # FLAGS.num_seqs,  # ？ 一个batch 的 句子 数目
                    num_steps=dh.max_document_length,  # FLAGS.num_steps,  # 一个句子的长度
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    embedding_weight=embedding_weight,
                    sampling=True,
                    dh=dh
                    )

    model.load(FLAGS.checkpoint_path)
    # cs = []
    # cs.append('♠是什么类型的产品')
    # cs.append('♠是谁')
    # cs.append('♠是哪个公司的长度')
    f1 = '../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt'
    f3 = '../data/nlpcc2016/4-ner/extract_entitys_all_tj.v1.txt'
    f4 = '../data/nlpcc2016/4-ner/extract_entitys_all_tj.sort_by_ner_lstm.v1.txt'
    f1s = ct.file_read_all_lines_strip(f1)
    f3s = ct.file_read_all_lines_strip(f3)
    f1s_new = []
    f3s_new = []
    for i in range(len(f1s)):
        # if str(f1s[i]).__contains__('NULL'):
        #     continue
        f1s_new.append(f1s[i])
        f3s_new.append(f3s[i])

    # 过滤NULL
    # 获取候选实体逐个去替代和判断

    # cs.append('立建候时么什是♠')
    # 读取出所有候选实体并打分取出前3 看准确率

    f4s = []
    _index = -1
    for l1 in f1s_new:  # 遍历每个问题
        _index += 1
        replace_qs = []
        for l3 in f3s_new[_index].split('\t'):
            q_1 = str(l1).split('\t')[0].replace(l3, '♠')
            replace_qs.append((q_1, l3))
        entitys = []
        for content, l3 in replace_qs:
            # content = input("input:")
            start = dh.convert_str_to_indexlist_2(content, False)

            # arr = model.sample(FLAGS.max_length, start, dh.converter.vocab_size,dh.get_padding_num())
            # #converter.vocab_size
            r1, score_list = model.judge(start, dh.converter.vocab_size)
            entitys.append((l3, r1))
            # print(content)
            # print(r1)
            # print(score_list)
            ct.print("%s\t%s\t%s" % (content, l3, r1), 'debug_process')
        entitys.sort(key=lambda x: x[1])
        entitys_new = [x[0] for x in entitys]
        ct.print('\t'.join(entitys_new))
        f4s.append('\t'.join(entitys_new))
    ct.file_wirte_list(f4, f4s)

    # print(arr)
    # print(dh.converter.arr_to_text_no_unk(arr))


if __name__ == '__main__':
    tf.app.run()
