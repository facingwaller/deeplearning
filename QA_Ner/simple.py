# 尝试使用简单的办法去做识别
# QA的NN
# author:ender
# date:2018.3.24


import tensorflow as tf
import lib.data_helper as data_helper
import QA.custom_nn as mynn
import time
import datetime
import numpy as np
from lib.ct import ct
from lib.config import FLAGS, get_config_msg
from lib.config import config
import os
from QA_GAN.QACNN import QACNN
from QA_GAN.Discriminator import Discriminator
from QA_GAN.Generator import Generator
from lib.ct import ct

if __name__ == "__main__":
    # print(config.get_model())
    print(1)
