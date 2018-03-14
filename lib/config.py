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

testid = "cc_test"
# ==正常调参

if testid == "cc_test":
    # 极限情况下调,1个问题，全关系
    epoches = 100  # 遍历多少轮
    batch_size = 10  # 1个batch的大小 # 临时改了
    evaluate_every = 4000  # 100训练X次验证一次   #等会临时改成20 - 10 试试看
    evaluate_batchsize = 2000  # 验证一次的问题数目
    questions_len_train = 4000  # 所有问题数目
    questions_len_test = 4000
    wrong_relation_num = 999999999999999  # 错误的关系，设置9999可以是全部的意思
    total_questions = 999999999999999
    stop_loss_zeor_count = 2000
    rnn_size = 100
    mode = "cc"
    check = 100000

    use_property = 'special'
    # 使用属性的模式做训练和测试
    # 1 num 限制数量 2 special 指定 3 no 非训练模式 4 maybe 模糊属性的单独处理
    skip_threshold = 0.02
    t_relation_num = 4000
    # 分割训练和测试 数据集的时候 使用正式的划分（严格区分训练和测试），
    # 而非模拟测试的。 之前是混合在一起
    real_split_train_test = True
    #####
    train_part = 'answer'  # 属性 relation |answer

else:
    epoches = 100 * 100 * 100  # 遍历多少轮
    batch_size = 10  # 1个batch的大小
    evaluate_every = 100
    evaluate_batchsize = 100
    questions_len_train = 800  # 应该设置大一点,2-10倍
    questions_len_test = int(questions_len_train / 4)
    wrong_relation_num = 99999  # 错误的关系，设置9999可以是全部的意思
    stop_loss_zeor_count = 10000
    rnn_size = 300
    mode = "sq"
    check = 999
    raise Exception("testid 参数有误")

# config.par('sq_fb_rdf_path')
sq_p = {
    'restore_path': '',
    #
    'sq_q_path': "../data/simple_questions/fb_0_2m_files/annotated_fb_data_all.txt-0.txt",
    'sq_fb_path': "../data/simple_questions/fb_0_2m_files/",
    'sq_fb_rdf_path': "../data/simple_questions/fb_0_2m_rdf",
    # ----------nplcc2016

    # ../data/nlpcc2016/ner_t1/q.rdf.m_s.txt
    'cc_path': '../data/nlpcc2016/demo1',  # nlpcc-iccpol-2016.kbqa.kb.out.txt
    'cc_kb_path_full': '../data/nlpcc2016/demo1/kb.txt',  # 使用的是简版的KB，只包含有答案的实体
    'cc_kb_path_big': '../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb.out.txt',  # 使用的是简版的KB，只包含有答案的实体
    'cc_kb_rdf_path': '',
    'cc_vocab': '../data/nlpcc2016/demo1/nlpcc2016.vocab',
    'cc_w2v': '',  # 词汇
    # 'baike_dict_name':'baike.dict', # 词汇
    'baike_dict_path': '../data/nlpcc2016/common/baike.dict'  # 词汇

}

cc_p = {
    # 'q.rdf': '../data/nlpcc2016/ner_t1/q.rdf.txt',  # 原始的
    'q.rdf.m_s': '../data/nlpcc2016/3-questions/q.rdf.m_s.txt',  # 带RDF，识别出的实体的RDF
    'q.rdf.m_s.filter': '../data/nlpcc2016/3-questions/q.rdf.m_s.filter.txt',  # 且过滤掉不能使用的方便查找
    # 'q.rdf.txt.math_s': '../data/nlpcc2016/ner_t1/q.rdf.txt.math_s.txt',  # 匹配到S的文件
    'q.final': '../data/nlpcc2016/3-questions/q.rdf.m_s.filter.suggest.txt',  # 带RDF，识别出的实体的RDF
    'rdf_extract_property': '../data/nlpcc2016/5-class/rdf_extract_property_origin.txt',  # 包含属性及其对应的index
    'rdf_extract_property_test': '../data/nlpcc2016/5-class/rdf_extract_property_origin_test.txt',  # 包含属性及其对应的index
    'rdf_maybe_property': '../data/nlpcc2016/5-class/maybe_possible.txt',
    'rdf_maybe_property_index': '../data/nlpcc2016/5-class/maybe_possible-index.txt',
    #
    'log_q_neg_r_tuple': '../data/nlpcc2016/5-class/log_q_neg_r_tuple.txt',
    # 知识库-完整版
    'kb': '../data/nlpcc2016/2-kb/kb.v1.txt',
    # 抽取所有可能答案的KB
    'kb.v3': '../data/nlpcc2016/2-kb/kb.v3.txt',
    # 所有正负例
    'q_neg_r_tuple': '../data/nlpcc2016/3-questions/q_neg_r_tuple.v1.txt',
    # 只包含使用的部分
    'kb-use': '../data/nlpcc2016/2-kb/kb-use.v2.txt',
    # 匹配实体和过滤后的问题集
    'cc_q_path': '../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',

    'real_split_train_test': True,
    'real_split_train_test_skip': 14610,
    'use_property':use_property,  # 记录进日志
    'train_part' : train_part,  # 属性 relation |answer
    'combine':'../data/nlpcc2016/9-combine/step.txt',
    'combine_test':'../data/nlpcc2016/9-combine/step_test.txt',

}

if questions_len_test < evaluate_batchsize:
    raise Exception("验证batch的size要大于总问题个数 %d <= %d"
                    %(questions_len_test,evaluate_batchsize))

# 模型
tf.flags.DEFINE_string("mode", mode, "是否增加attention机制 ")
tf.flags.DEFINE_boolean("need_cal_attention", True, "是否增加attention机制 ")
tf.flags.DEFINE_boolean("need_max_pooling", False, "是否增加max_pooling机制 ")
tf.flags.DEFINE_boolean("need_test", True, "是否测试 ")
tf.flags.DEFINE_boolean("fix_model", True, "是否开启纠错模式 ")

tf.flags.DEFINE_integer("questions_len_train", questions_len_train, "questions_len_train  ")
tf.flags.DEFINE_integer("questions_len_test", questions_len_test, "questions_len_test  ")
tf.flags.DEFINE_integer("t_relation_num", t_relation_num, "t_relation_num 关系数量  ")

tf.flags.DEFINE_integer("total_questions", total_questions, "总共的问题数  ")
# 训练-验证-测试
tf.flags.DEFINE_integer("epoches", epoches, "epoches")
# tf.flags.DEFINE_integer("num_classes", 100, "num_classes 最终的分类")
# tf.flags.DEFINE_integer("num_hidden", 100, "num_hidden 隐藏层的大小")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding_size")
tf.flags.DEFINE_integer("rnn_size", rnn_size, "LSTM 隐藏层的大小 ")
tf.flags.DEFINE_integer("batch_size", batch_size, "batch_size")
tf.flags.DEFINE_integer("max_grad_norm", 5, "max_grad_norm")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_integer("check", check, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", evaluate_every, "evaluate_every")
tf.flags.DEFINE_integer("evaluate_batchsize", evaluate_batchsize, "test_batchsize ")
tf.flags.DEFINE_integer("test_every", evaluate_every, "test_every ")
tf.flags.DEFINE_integer("test_batchsize", evaluate_batchsize, "test_batchsize ")

tf.flags.DEFINE_integer("stop_loss_zeor_count", stop_loss_zeor_count, "loss=0 停止的次数 ")
ms = ["train", "test"
    , "debug"
    , "none"
    , "time"
    , "show_shape"
    , "data"
    , "debug_epoches"
    , "bad"
      ]


def get_config_msg():
    FLAGS._parse_flags()
    FLAGS_Parameters = "\nParameters:\n"
    for attr, value in sorted             (FLAGS.__flags.items()):
        FLAGS_Parameters += "{}={}\n".format(attr.upper(), value)
    for item in cc_p:
        FLAGS_Parameters += '%s\n' % cc_p[item]

    return FLAGS_Parameters


myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)


## 配置根据机器、想运行的模式等决定
##

class config:
    @staticmethod
    def get_find_index():
        return 13335

    @staticmethod
    def par(key):
        return sq_p[key]

    @staticmethod
    def cc_par(key):
        return cc_p[key]

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

    # @staticmethod
    # def get_sq_topic_path():
    #     return '../data/simple_questions/fb_0'

    # @staticmethod
    # def get_sq_files_path():
    #     return '../data/simple_questions/fb_0_files'

    # 使用极少数条数据做测试
    @staticmethod
    def is_debug_few():
        return True

    @staticmethod
    def get_static_id_list_debug(max_length):

        q_l_t = min(max_length, questions_len_train)
        a = []
        for i in range(0, q_l_t - 1):
            a.append(i)
        return a

    @staticmethod
    def get_static_id_list_debug_test(max_length):
        q_l_t = min(max_length, questions_len_test)
        a = []
        # random.randint() 考虑改成随机的10个
        # a <= n <= b
        # min,max = 1,1000
        for i in range(0, q_l_t ):
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
            filename = config.par('sq_fb_path') + "/wiki.vector"  # '../data/simple_questions/fb_0_files/wiki.vector'
        elif mode == "cc":
            filename = config.par('cc_path') + "/wiki.vector"  # '../data/simple_questions/fb_0_files/wiki.vector'
        else:
            print("???")
            raise Exception("mode error")
        return filename

    @staticmethod
    def get_print_type():
        return ms

    @staticmethod
    def get_total_questions():
        return total_questions

    @staticmethod
    def get_t_relation_num():
        return t_relation_num

    @staticmethod
    def use_property():
        return use_property

    @staticmethod
    def skip_threshold():
        return skip_threshold


if __name__ == "__main__":
    print(config.cc_par('train_part'))
    # config.test()
