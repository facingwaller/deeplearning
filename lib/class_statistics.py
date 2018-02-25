# 如何分辨属性分布，分辨统计和对比两份数据集里面的属性的交集，差集，测试里面有的，
# 训练里面没有的。
from lib.ct import ct


class class_stat:
    @staticmethod
    def rel_statistics():
        flag = 14254  # 从这个开始是测试集
        # 1 读取2份数据
        # 2 读取对应的
        f1s = ct.file_read_all_lines_strip('../data/nlpcc2016/class/q.rdf.m_s.filter.txt')

        f1s_train = f1s[0:flag]
        f1s_test = f1s[flag:]
        r_train = set()
        r_test = set()

        for item in f1s_train:
            if len(str(item).split('\t')) < 4:
                continue
            r_train.add(str(item).split('\t')[3].lower().replace(' ', ''))
        for item in f1s_test:
            if len(str(item).split('\t')) < 4:
                continue
            r_test.add(str(item).split('\t')[3].lower().replace(' ', ''))

        # 得到指定数据集里面的全部属性
        r3 = (r_train | r_test) - r_train
        for r in r3:
            ct.just_log("../data/nlpcc2016/class/q.rdf.m_s.filter[r_in_test_not_in_train].txt", r)
        print(1)

    @staticmethod
    def ner_re_writer(f1='../data/nlpcc2016/ner_t1/q.rdf.m_s.filter.txt',
                      f2='../data/nlpcc2016/class/q.rdf.m_s.filter.re_writer.txt'):
        """
        重写问句库
        """
        # 1. 读取问句库
        # 2. 替换问句并输出

        f1s = ct.file_read_all_lines_strip(f1)
        f1s_new = []
        for f1s_l in f1s:
            s1 = str(f1s_l).split('\t')
            e1 = str(f1s_l).split('\t')[5]
            q1 = str(f1s_l).split('\t')[0].replace(' ','').lower()
            # .replace('','♠')
            q2 = str(q1).replace(e1, '♠')
            s1.append(q2)
            f1s_new.append('\t'.join(s1))
        ct.file_wirte_list(f2, f1s_new)

        print(1)


if __name__ == "__main__":
    class_stat.ner_re_writer()
