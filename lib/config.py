"""
config all par
配置所有的参数。
author:ender
data:2018.1.22
♠黑桃 ♢方块
"""
import socket
import tensorflow as tf
from enum import Enum

class optimizer_m(Enum):
    lstm = 1
    gan = 2


FLAGS = tf.flags.FLAGS

# myname = socket.getfqdn(socket.gethostname())
# myaddr = socket.gethostbyname(myname)
# # embedding

# tf_embedding tf自己训练
# word2vec_train 用word2vec重新组建W矩阵，并随后更新

# testid = "cc_test"
# if str(myaddr) == "192.168.31.194":
#     testid = "cc_test"
#     cmd_path = r'F:\ProgramData\Anaconda2\envs\tensorflow\python.exe F:/PycharmProjects/dl2/deeplearning/QA/train.py'
# else:
#     testid = "cc_test"
#     cmd_path = r'C:\ProgramData\Anaconda2\envs\tensorflow\python.exe C:/Users/flow/PycharmProjects/tensorFlow1/QA/train.py'
# print("%s\t%s" % (myaddr, testid))

# ==正常调参


# 配置说明  命名实体识别方案
def s_config():
    assert t_relation_num == 4385 # 所有属性对应的问题
    assert d_epoches == 1 # 使用D模式训练
    assert train_part == 'entity'  # 训练的是实体(指示训练的模块、以及计算得分的模块)
    assert ns_model == 'entity'  # negative sampling 使用产生负例实体
    assert ner_top_cand == 2 # 指示取前3个（2个负例，1个正例）做训练

# 配置说明  属性识别方案
def p_config():
    assert t_relation_num == 4385  # 所有属性对应的问题
    assert d_epoches == 1   # 使用D模式训练
    assert ns_epoches == 1
    assert train_part == 'relation'  # 指示训练的模块、以及计算得分的模块
    assert ns_model == 'only_default'  # negative sampling 使用产生负例实体；属性是重点，使用了多种方法。
    assert ner_top_cand == 0    # 指示 取0个命名实体负例，

def er_config():
    assert t_relation_num == 4385  # 所有属性对应的问题
    assert ner_epoches == 1   # 使用D模式训练
    assert ns_epoches == 1  # 使用NS模式训练
    assert train_part == 'entity_relation'  # 属性 relation |answer | entity  |  entity_relation
    assert loss_part == 'entity_relation'  # entity_relation ；entity_relation_transE;entity;relation

def er_cp_config():
    assert t_relation_num == 4385  # 所有属性对应的问题
    assert ner_epoches == 1   # 使用D模式训练
    assert ns_epoches == 1  # 使用NS模式训练
    assert train_part == 'entity_relation'  # 属性 relation |answer | entity  |  entity_relation
    assert loss_part == 'entity_relation'  # entity_relation ；entity_relation_transE;entity;relation

def ert_cp_config():
    assert t_relation_num == 4385  # 所有属性对应的问题
    assert ner_epoches == 1   # 使用D模式训练
    assert ns_epoches == 1  # 使用NS模式训练
    assert train_part == 'entity_relation'  # 属性 relation |answer | entity  |  entity_relation
    assert loss_part == 'entity_relation_transE'  # entity_relation ；entity_relation_transE;entity;relation

# =========================================== 基本不需要调整
epoches = 100   # 遍历多少轮
evaluate_every = 100  # 100训练X次验证一次   #等会临时改成20 - 10 试试看
evaluate_batchsize = 999999  # 验证一次的问题数目,超过则使用最大的
questions_len_train = 99999999999  # 所有问题数目
questions_len_test = 999999999999  # 测试的问题数目，全部
wrong_relation_num = 999999999999  # 错误的关系，设置9999可以是全部的意思
total_questions = 99999999
stop_loss_zeor_count = 2000  # 2000次则停下来
rnn_size = 100
mode = "cc" #cc 中文训练  cc_test 中文测试
check = 100000
# 属性模式 # 1 num 限制数量 2 special 指定 3 no 非训练模式 4 maybe 模糊属性的单独处理
use_property = 'special'  # 使用属性的模式做训练和测试
# skip_threshold = 0.05
pre_train = True

# ################## 可能调整
loss_margin = 0.05  # 简书上是0.05，liu kang 那边是 0.6 也有0.02
attention_model = 'a_side'  # a_side (默认) 问题端或者答案端  q_side  both
batch_size = 5  # 1个batch的大小 # 临时改成1 个看loss

# ==================================需要配置
# mark = '测试CP的效果；寻找最佳的CP策略；10P;NEG负例的个数是全部 ;14-cp的竞争策略[20181013-1]' # 备注
mark = '测试CP的效果；' # 备注
t_relation_num = 100  # 4385  # 重要！这个指示了训练的关系个数 4358
ner_top_cand = 0  # 训练取2，测试取3（写死） ; 0 只测属性识别,2 测实体或者实体+属性

real_split_train_test = True  # 严格区分训练和测试
train_part = 'relation'  # 属性 relation |answer | entity  |  entity_relation
loss_part = '...'  # entity_relation ；entity_relation_transE;entity;relation
# entity_relation_transE|  entity_relation_answer_transE |  relation | entity | transE
#  IR-GAN
ns_model = 'competing_q'  # competing_q_ert competing_q  only_default only_default_er  random entity temp_test_all
g_epoches = 0
d_epoches = 1
s_epoches = 0
c_epoches = 0
a_epoches = 0
ner_epoches = 0
ns_epoches = 1

# ===============================实验的时候才调整
batch_size_gan = 200  # 100 或者 1000 ,80%的竞争属性是在600
gan_k = 10
sampled_temperature = 20
gan_learn_rate = 0.05
d_need_cal_attention = True
g_need_cal_attention = True
competing_s_neg_p_num = 10   # 竞争属性中，P_POS的负例的多少 用于 ert
competing_p_pos_neg_size = 9999  # 竞争属性中，P_POS的负例的多少
convert_rs_to_words = False   # 关系集合转换为对应的字符集合
only_p_neg_in_cp = False               # 只加入P_NEG进入 CP 暂停，这样会导致POS无法加入
hand_add_some_neg = False
does_cp_contains_default = False # competing_q 模式下 是否包含默认的 属性。去掉才显得出
# only_default 默认模式|fixed_amount 固定 最多100个 | additional 默认+额外
# synonym_train_mode 优先加入neg的同义词
# competing_ps 竞争属性
# pool_mode = 'additional' # competing_ps | None | additional | only_default

# 一些调整的参数
ns_ps_try_only_pos = False
ns_ps_len_max_limit = 22
ns_q_ploicy_all = 'all_p' # D的策略，全问题还是一个 all_p , 1_q,1_p
# 模型恢复
restore_model = True
restore_path_base = r'F:\PycharmProjects\dl2\deeplearning\QA_GAN\runs'
restore_path = restore_path_base+ \
               r'\2018_11_05_20_13_54_100p_retest\checkpoints\step=1_epoches=d_index=0\model.ckpt-1'
                 # r'\2018_10_26_10_58_29_100p_default\checkpoints\step=3_epoches=d_index=0\model.ckpt-1'
#                  r'\2018_11_01_11_45_50_allp_default_1\checkpoints\step=1_epoches=d_index=0\model.ckpt-1'
# r'\2018_11_05_16_45_17_100p_cp_4\checkpoints\step=21_epoches=d_index=0\model.ckpt-1'
                #r'\2018_10_05_11_06_59_allp_ns_cp_best\checkpoints\step=6_epoches=d_index=0\model.ckpt-1'
restore_test = False


# 竞争属性集
# competing_ps_path = '../data/nlpcc2016/5-class/competing_ps.v1.txt'
competing_ps_path = '../data/nlpcc2016/14-cp/competing_ps_tj.v2.txt'
# competing_ps_path = '../data/nlpcc2016/13-competing/competing_ps_tj.v2.txt'# 13-competing 版本的
# competing_ps_path = '../data/nlpcc2016/13-competing/competing_p_in_kb.v2.txt'
competing_batch_size = 10 # 控制size 无用

expend_es = '../data/nlpcc2016/4-ner/result/q.rdf.score.top_3_all_0.v4.10.txt'

# S-NER
ner_model = 'cos'
ner_path = '../data/nlpcc2016/4-ner/extract_entitys_all_tj_sort.v1.txt'

alias_dict = '../data/nlpcc2016/4-ner/extract_e/e1.dict.txt'
use_alias_dict = True
# 负例选取的方式 alias 指代后的全体实体的属性 | competing 某POS属性的竞争属性
negative_sampling_model = 'alias'  # alias | competing


# 只验证错误的模式 only_error|all
valid_model = 'all'
valid_only_error_valid = '../data/nlpcc2016/7-error/only_error/valid.v1.txt'
valid_only_error_test = '../data/nlpcc2016/7-error/only_error/test.v1.txt'
keep_run = False  # 指示是否持续跑maybe里面的属性
optimizer_method = optimizer_m.gan  # 优化模式 gan | lstm
synonym_mode = 'none'  # 属性同义词 ps_synonym| none
synonym_train_mode = 'none'  # 同义词的训练模式 synonym_train_mode|none
# synonym S_model
S_model = 'none'  # S_model | none


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
    'cc_q_path': '../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt',
    # q.rdf.score.expend.v1.txt
    'real_split_train_test': True,
    'real_split_train_test_skip': 14097, # 14610
    'real_split_train_test_skip_v2': 14097,
    'use_property': use_property,  # 记录进日志
    'train_part': train_part,  # 属性 relation |answer
    'loss_part': loss_part,  # 属性 relation |answer
    'loss_margin': loss_margin,
    'combine': '../data/nlpcc2016/9-combine/step.txt',
    'combine_test': '../data/nlpcc2016/9-combine/step_test.txt',
    'test_ps': '../data/nlpcc2016/5-class/test_ps.txt',
    'test_ps_result': '../data/nlpcc2016/5-class/test_ps_result.txt',
    # 'cmd_path': cmd_path,
    'keep_run': keep_run,
    'optimizer_method': optimizer_method,
    'mark': mark,
    'gan_learn_rate': gan_learn_rate,
    # 'pool_mode': pool_mode,
    #
    'restore_model': restore_model,
    'restore_path': restore_path,
    'restore_test':restore_test,
    #
    'synonym_mode': synonym_mode,
    'synonym_words': '../data/nlpcc2016/5-class/synonym/same_p_tj_score.v2.3.txt',
    'synonym_train_data': '../data/nlpcc2016/5-class/synonym/all/same_p_tj_clear_dict.txt',
    # 'synonym_train_data': '../data/nlpcc2016/5-class/synonym/all/tongyici_dict_1.txt',
    'synonym_train_mode': synonym_train_mode,
    'S_model': S_model,
    'valid_model': valid_model,
    'valid_only_error_valid': valid_only_error_valid,
    'valid_only_error_test': valid_only_error_test,
    # 竞争
    'competing_ps_path': competing_ps_path,
    'competing_batch_size':competing_batch_size,
    'competing_s_neg_p_num':competing_s_neg_p_num,
    ## NER
    'expend_es':expend_es,
    'ner_model' : ner_model,
    'ner_path' : ner_path,
    'ner_top_cand':ner_top_cand,
    't_relation_num':t_relation_num,
    'alias_dict':alias_dict,
    'use_alias_dict':use_alias_dict,
    'negative_sampling_model':negative_sampling_model,
    'pre_train':pre_train,
    # 注意力
    'attention_model':attention_model,
    'd_need_cal_attention':d_need_cal_attention,
    'g_need_cal_attention':g_need_cal_attention,
    'ns_model':ns_model,
    'competing_p_pos_neg_size':competing_p_pos_neg_size,
    'convert_rs_to_words':convert_rs_to_words,
    'only_p_neg_in_cp':only_p_neg_in_cp,
    'hand_add_some_neg':hand_add_some_neg,
    'does_cp_contains_default':does_cp_contains_default,
    'ns_ps_len_max_limit':ns_ps_len_max_limit,
    'ns_ps_try_only_pos':ns_ps_try_only_pos,
    'ns_q_ploicy_all':ns_q_ploicy_all

}


# 模型
tf.flags.DEFINE_integer("word_dimension", 100, "单词的维度 ")
tf.flags.DEFINE_string("word_model", "word2vec_train", "可选有|tf_embedding|word2vec_train|word2vec")
tf.flags.DEFINE_string("mode", mode, "是否增加attention机制 ")
# tf.flags.DEFINE_boolean("need_cal_attention", need_cal_attention, "是否增加attention机制 ")
tf.flags.DEFINE_boolean("need_max_pooling", False, "是否增加max_pooling机制 ")
tf.flags.DEFINE_boolean("need_test", True, "是否测试 ")
tf.flags.DEFINE_boolean("fix_model", True, "是否开启纠错模式 ")

tf.flags.DEFINE_integer("questions_len_train", questions_len_train, "questions_len_train  ")
tf.flags.DEFINE_integer("questions_len_test", questions_len_test, "questions_len_test  ")
tf.flags.DEFINE_integer("t_relation_num", t_relation_num, "t_relation_num 关系数量  ")

tf.flags.DEFINE_integer("total_questions", total_questions, "总共的问题数  ")
# 训练-验证-测试
tf.flags.DEFINE_integer("epoches", epoches, "epoches")
tf.flags.DEFINE_integer("g_epoches", g_epoches, "g_epoches")
tf.flags.DEFINE_integer("d_epoches", d_epoches, "d_epoches")
tf.flags.DEFINE_integer("s_epoches", s_epoches, "s_epoches")
tf.flags.DEFINE_integer("c_epoches", c_epoches, "c_epoches")
tf.flags.DEFINE_integer("a_epoches", a_epoches, "a_epoches")
tf.flags.DEFINE_integer("ner_epoches", ner_epoches, "ner_epoches")
tf.flags.DEFINE_integer("ns_epoches", ns_epoches, "ns_epoches")
# ns_epoches

tf.flags.DEFINE_integer("embedding_size", 100, "embedding_size")
tf.flags.DEFINE_integer("rnn_size", rnn_size, "LSTM 隐藏层的大小 ")
tf.flags.DEFINE_integer("batch_size", batch_size, "batch_size")
tf.flags.DEFINE_integer("batch_size_gan", batch_size_gan, "batch_size_gan")
tf.flags.DEFINE_integer("max_grad_norm", 5, "max_grad_norm")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_integer("check", check, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", evaluate_every, "evaluate_every")
tf.flags.DEFINE_integer("evaluate_batchsize", evaluate_batchsize, "test_batchsize ")
tf.flags.DEFINE_integer("test_every", evaluate_every, "test_every ")
tf.flags.DEFINE_integer("test_batchsize", evaluate_batchsize, "test_batchsize ")

tf.flags.DEFINE_integer("stop_loss_zeor_count", stop_loss_zeor_count, "loss=0 停止的次数 ")
tf.flags.DEFINE_integer("gan_k", gan_k, "生成 FLAGS.gan_k个负例  ")
tf.flags.DEFINE_integer("sampled_temperature", sampled_temperature, "the temperature of sampling")
tf.flags.DEFINE_integer("gan_learn_rate", gan_learn_rate, "gan_learn_rate  ")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_boolean("restore_model", restore_model, "Log placement of ops on devices")


ms = ["train", "test"
    , "debug"
    , "none"
    , "time"
    , "show_shape"
    , "data"
    , "debug_epoches"
    , "bad"
    , "loss"
      ,'cost'
      # expection
      ]


def get_config_msg():
    FLAGS._parse_flags()
    FLAGS_Parameters = "\nParameters:\n"
    for attr, value in sorted(FLAGS.__flags.items()):
        FLAGS_Parameters += "{}={}\n".format(attr.upper(), value)
    for item in cc_p:
        FLAGS_Parameters += '%s\n' % cc_p[item]

    return FLAGS_Parameters


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
    def cc_compare(key, value):
        return cc_p[key] == value

    # @staticmethod
    # def get_model():
    #     if str(myaddr) == "192.168.31.194":
    #         return "test"
    #     else:
    #         return "train"
    #
    # @staticmethod
    # def get_config_path():
    #
    #     # print(myaddr)
    #     # F:\3_Server\freebase-data\topic-json
    #     if str(myaddr) == "192.168.31.194":
    #         return r"F:\3_Server\freebase-data\topic-json"
    #     else:
    #         return r"D:\ZAIZHI\freebase-data\topic-json"

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
        for i in range(0, q_l_t):
            a.append(i)
        return a

    @staticmethod
    def get_static_id_list_debug_test(max_length):
        q_l_t = min(max_length, questions_len_test)
        a = []
        # random.randint() 考虑改成随机的10个
        # a <= n <= b
        # min,max = 1,1000
        for i in range(0, q_l_t):
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
        elif mode == "ner":
            filename = config.par('cc_path') + "/wiki.vector"
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




if __name__ == "__main__":
    # print(config.get_model())
    print(restore_path)
    ns_q_state_all= []
    global_index = 1
    import time
    time_start = time.time()

    print('cost %d/%d: %s' % (global_index, len(ns_q_state_all), time.time() - time_start))
    s1 = "《点石成金》是谁写的？".replace('点石成金','???')
    print(s1)
    print(config.cc_par('train_part'))
    print(use_property)
