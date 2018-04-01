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


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    model = 'ner'
    dh = data_helper.DataClass(model)
    train_batch_size = 2
    # g = dh.batch_iter_char_rnn(train_batch_size)  # (FLAGS.num_seqs, FLAGS.num_steps)
    embedding_weight = dh.embeddings
    # with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
    #     text = f.read()
    # converter = TextConverter(text, FLAGS.max_vocab)
    # converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
    #
    # arr = converter.text_to_arr(text)
    # g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)

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
                    sampling=False,
                    dh=dh
                    )

    model.train(
        FLAGS.max_steps,
        model_path,
        FLAGS.save_every_n,
        FLAGS.log_every_n,
    )


if __name__ == '__main__':
    tf.app.run()
