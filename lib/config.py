"""
config all par
配置所有的参数。
author:ender
data:2018.1.22
"""
import socket
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# embedding
tf.flags.DEFINE_integer("word_dimension", 100, "单词的维度 ")
# tf_embedding tf自己训练
# word2vec_train 用word2vec重新组建W矩阵，并随后更新
tf.flags.DEFINE_string("word_model", "word2vec_train", "可选有|tf_embedding|word2vec_train|word2vec")

testid = 1
# ==正常调参
if testid == 100:
    epoches = 100 * 100*100  # 遍历多少轮
    batch_size = 10  # 1个batch的大小
    evaluate_every = 100
    evaluate_batchsize = 100
    questions_len_train = 800  # 应该设置大一点,2-10倍
    questions_len_test = int(questions_len_train / 4)
    wrong_relation_num = 99999  # 错误的关系，设置9999可以是全部的意思
    stop_loss_zeor_count = 10000
    rnn_size = 300
elif testid == 1:
    # 极限情况下调,1个问题，全关系
    epoches = 100  # 遍历多少轮
    batch_size = 10  # 1个batch的大小
    evaluate_every = 60  # 100训练X次验证一次
    evaluate_batchsize = batch_size * 5  # 验证一次的问题数目
    questions_len_train = batch_size * 5  # 所有问题数目
    questions_len_test = questions_len_train
    wrong_relation_num = 9999  # 错误的关系，设置9999可以是全部的意思
    stop_loss_zeor_count = 2000
    rnn_size = 100
elif testid == 0:
    # 极限情况下调,1个问题，全关系
    epoches = 100  # 遍历多少轮
    batch_size = 10  # 1个batch的大小
    evaluate_every = 60  # 100训练X次验证一次
    evaluate_batchsize = batch_size * 5  # 验证一次的问题数目
    questions_len_train = 99999  # 所有问题数目
    questions_len_test = questions_len_train
    wrong_relation_num = 9999  # 错误的关系，设置9999可以是全部的意思
    stop_loss_zeor_count = 2000
    rnn_size = 100
else:
    print("?????")

# 模型
tf.flags.DEFINE_boolean("need_cal_attention", True, "是否增加attention机制 ")
tf.flags.DEFINE_boolean("need_max_pooling", False, "是否增加max_pooling机制 ")
tf.flags.DEFINE_boolean("need_test", True, "是否测试 ")
tf.flags.DEFINE_boolean("fix_model", True, "是否开启纠错模式 ")

tf.flags.DEFINE_integer("questions_len_train", questions_len_train, "questions_len_train  ")
tf.flags.DEFINE_integer("questions_len_test", questions_len_test, "questions_len_test  ")


# 训练-验证-测试
tf.flags.DEFINE_integer("epoches", epoches, "epoches")
# tf.flags.DEFINE_integer("num_classes", 100, "num_classes 最终的分类")
# tf.flags.DEFINE_integer("num_hidden", 100, "num_hidden 隐藏层的大小")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding_size")
tf.flags.DEFINE_integer("rnn_size", rnn_size, "LSTM 隐藏层的大小 ")
tf.flags.DEFINE_integer("batch_size", batch_size, "batch_size")
tf.flags.DEFINE_integer("max_grad_norm", 5, "max_grad_norm")
tf.flags.DEFINE_integer("num_checkpoints", 5000, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_integer("check", 10000, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", evaluate_every, "evaluate_every")
tf.flags.DEFINE_integer("evaluate_batchsize", evaluate_batchsize, "test_batchsize ")
tf.flags.DEFINE_integer("test_every", evaluate_every, "test_every ")
tf.flags.DEFINE_integer("test_batchsize", evaluate_batchsize, "test_batchsize ")


tf.flags.DEFINE_integer("stop_loss_zeor_count", stop_loss_zeor_count, "loss=0 停止的次数 ")
ms = ["train", "test"
    , "debug"
      , "none"
    ,"time"
    , "show_shape"
      , "data"
      , "debug_epoches"
      ]


def get_config_msg():
    FLAGS._parse_flags()
    FLAGS_Parameters = "\nParameters:"
    for attr, value in sorted(FLAGS.__flags.items()):
        FLAGS_Parameters += "{}={}\n".format(attr.upper(), value)

    return FLAGS_Parameters


myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)


## 配置根据机器、想运行的模式等决定
##

class config:
    @staticmethod
    def get_model():
        if str(myaddr) == "192.168.31.194":
            return "test"
        else:
            return "train"

    @staticmethod
    def get_config_path():

        # print(myaddr)
        # F:\3_Server\freebase-data\topic-json
        if str(myaddr) == "192.168.31.194":
            return r"F:\3_Server\freebase-data\topic-json"
        else:
            return r"D:\ZAIZHI\freebase-data\topic-json"

    @staticmethod
    def get_sq_topic_path():
        return '../data/simple_questions/fb_1000/'
    # 使用极少数条数据做测试
    @staticmethod
    def is_debug_few():
        return True

    @staticmethod
    def get_static_id_list_debug():
        a = []
        for i in range(1, questions_len_train + 1):
            a.append(i)
        return a

    @staticmethod
    def get_static_id_list_debug_test():
        a = []
        # random.randint() 考虑改成随机的10个
        # a <= n <= b
        # min,max = 1,1000
        for i in range(1, questions_len_test + 1):
            a.append(i)
        return a

    # --获取指定个数的问题
    @staticmethod
    def get_static_q_num_debug():
        return questions_len_train

    # --获取指定个数的错误关系
    @staticmethod
    def get_static_num_debug():
        return wrong_relation_num

    @staticmethod
    def test():
        print(config.get_static_id_list_debug())

    @staticmethod
    def wiki_vector_path(mode):
        if mode == "wq":
            filename = '../data/word2vec/wiki.vector'
        elif mode == "sq":
            filename = '../data/simple_questions/fb_1000_files/wiki.vector'
        else:
            print("???")
            raise Exception("mode error")
        return filename

    @staticmethod
    def get_print_type():
        return ms


if __name__ == "__main__":
    config.test()
