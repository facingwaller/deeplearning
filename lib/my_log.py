
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import rnn_lstm.data_helpers as data_helpers
from tensorflow.contrib import learn
from tensorflow.python import debug as tfdbg
import numpy as np
import os
import time
import datetime
import pickle
import logging
import logging.handlers
# from gensim.models import word2vec
# from gensim import models

''' 日志  '''
LOG_FILE = 'log2/' + str(time.time()) + '.txt'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('tst')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)
logger.info('==================================')

def log_list(obj):
    logger.info("obj begin===========================" + str(len(obj)))
    for l1 in obj:
        # print(l1)
        logger.info(l1)
def get_logger():
    return logger
