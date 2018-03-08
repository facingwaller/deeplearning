from lib.baike_helper import baike_helper, baike_test, classification
from lib.data_helper import DataClass
from lib.config import config

# 一键纠错
if __name__ == '__main__':  #
    bkt = baike_test()
    bkh = baike_helper()
    cf = classification()
    if False:
        # 1 过滤KB
        baike_helper.clean_baike_kb(file_name="../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.kb",
                                    file_out_name="../data/nlpcc2016/2-kb/kb.v1.txt",
                                    clean_log_path="../data/nlpcc2016/2-kb/clean_baike_kb.txt")
        print('过滤KB')

        # 2 生成KB的实体统计文件,这个还不够，还需要结合em_by_1 em_by_2
    if False:
        baike_helper.statistics_subject_len(f_in="../data/nlpcc2016/2-kb/kb.v1.txt"
                                            ,
                                            f_out="../data/nlpcc2016/2-kb/kb-entity.v1.txt")
        print('NER部分 统计KB长度')

    if False:
        # 暂时不理这边NER的问题
        num = 999
        bkt.try_test_acc_of_m1(
            f1='../data/nlpcc2016/3-questions/q.rdf.txt',
            f3='../data/nlpcc2016/4-ner/extract_entitys_all_tj.txt',
            # extract_entitys_v3                extract_entitys_all
            f2='../data/nlpcc2016/4-ner/q.rdf.txt.failed_v3_%d.txt' % num,
            use_cx=False, use_expect=False, acc_index=[num],
            get_math_subject=True,
            f6='../data/nlpcc2016/4-ner/extract_entitys_all_tj.txt.statistics.txt',
            f7='../data/nlpcc2016/3-questions/q.rdf.txt.math_s.txt')
        print('重写math_s')
    if False:
        # 合并 q.rdf.txt.math_s.txt ， q.rdf 到 q.rdf.m_s
        bkh.rewrite_rdf(f3='../data/nlpcc2016/3-questions/q.rdf.txt',
                        f2='../data/nlpcc2016/3-questions/q.rdf.m_s.txt',
                        f1='../data/nlpcc2016/3-questions/q.rdf.txt.math_s.txt')
        print('重写q.rdf.m_s.txt')
    if False:
        # 重新选择一遍属性
        bkh.choose_property(f1='../data/nlpcc2016/3-questions/q.rdf.m_s.txt',
                            f2='../data/nlpcc2016/3-questions/q.rdf.m_s.suggest.txt')
    # 重写rdf_extract_property_origin
    # C1.2.1
    if False:
        cf.extract_property(f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.txt',
                            f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                            f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin.txt',
                            skip=0)
        print('重写q.rdf.ms.re.v1.filter.txt和rdf_extract_property_origin.txt')
    if False:
        # 仅用于测试
        cf.extract_property(f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                            f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter_test.txt',
                            f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin_test.txt',
                            skip=14610)

    # 根据答案抽取出精简的KB
    if False:
        # F0.1.3
        bkh.extract_kb_possible(f1='../data/nlpcc2016/2-kb/kb.v1.txt',
                                f2="../data/nlpcc2016/2-kb/kb-use.v2.txt",
                                f3='../data/nlpcc2016/3-questions/q.rdf.m_s.filter.txt')
        print('根据答案抽取出精简的KB kb-use.v2.txt')
    # 重写q.txt        # 3 生成新的训练文件
    if True:
        dh = DataClass(mode="cc", run_type='init')
        dh.build_all_q_r_tuple(99999999999999,
                               99999999999999, is_record=True)
        print('重新生成训练文件q_neg_r_tuple.v1')
