from lib.baike_helper import baike_helper, baike_test
from lib.classification_helper import classification
from lib.data_helper import DataClass
from lib.config import config

# 一键纠错
if __name__ == '__main__':  #
    bkt = baike_test()
    bkh = baike_helper()
    cf = classification()

    # -------------- 清理部分 NER部分
    if False:
        print('1 过滤KB')
        baike_helper.clean_baike_kb(file_name="../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.kb",
                                    file_out_name="../data/nlpcc2016/2-kb/kb.v1.1.txt",
                                    clean_log_path="../data/nlpcc2016/2-kb/clean_baike_kb.v1.1.txt")

    if False:
        print('2 生成KB的实体统计文件')
        # 初步抽取
        # bkh.extract_e(f1='../data/nlpcc2016/2-kb/kb.v1.txt',
        #           f2='../data/nlpcc2016/4-ner/extract_e/e1.txt')
        # 过滤一遍，抽取出别名
        # bkh.extract_alis_1(f1='../data/nlpcc2016/4-ner/extract_e/e1.txt',
        #                  f2='../data/nlpcc2016/4-ner/extract_e/e1.alis.txt',
        #                  f3='../data/nlpcc2016/4-ner/extract_e/e1.double.txt',
        #                  f4='../data/nlpcc2016/4-ner/extract_e/e1.dict.txt',
        #                  max_len=60)
        # 构建字典
        bkh.extract_alis_2(f1='../data/nlpcc2016/4-ner/extract_e/e1.alis.txt',
                           f4='../data/nlpcc2016/4-ner/extract_e/e1.dict.txt',
                           f5='../data/nlpcc2016/4-ner/extract_e/e1.set.txt')
        # 2 生成KB的实体统计文件,这个还不够，还需要结合em_by_1 em_by_2
    if False:
        print('3 计算长度然后排序')
        bkh.statistics_subject_len(f_in='../data/nlpcc2016/4-ner/extract_e/e1.set.txt',
                                   f_out='../data/nlpcc2016/4-ner/extract_e/e1.tj.txt')
        # bkh.tmp_compare_1()
        print('NER部分 统计KB长度')
        # 计算IDF
    if False:
        baike_test.try_idf(f1='../data/nlpcc2016/4-ner/extract_entitys_all_tj.txt',
                           f2='../data/nlpcc2016/4-ner/extract_entitys_all.txt.statistics.txt',
                           f3='../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt',
                           skip=14610)
    if False:

        # 重写extract_entitys_all_tj 过滤不存在的属性
        bkh.filter_not_exist_in_f11(f1='../data/nlpcc2016/4-ner/extract_e/e1.tj.txt',
                                    f2='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v0.txt',
                                    f3='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v1.txt')
        # bkh.init_find_entity()
        # bkh.init_ner(f11)  # bkh.n_gram_dict[time] = word list

    if False:
        num = 99
        filter_list1 = bkt.try_test_acc_of_m1(
            f1='../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt',
            f3='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v1.txt',
            # extract_entitys_v3                extract_entitys_all
            f2='../data/nlpcc2016/4-ner/demo2/q.rdf.txt.failed_v4.8_%d.txt' % num,
            use_cx=False, use_expect=False, acc_index=[num],
            get_math_subject=True,
            f6='../data/nlpcc2016/4-ner/extract_entitys_all.txt.statistics.txt',
            f8='../data/nlpcc2016/4-ner/demo2/extract_entitys_all_tj.resort_%d.v4.8.txt' % num,
            f9='../data/nlpcc2016/4-ner/demo2/q.rdf.ms.re.top_%d.v4.10.txt' % num,
            f10='../data/nlpcc2016/4-ner/demo2/ner_%d.v4.10.txt' % num,
            combine_idf=True,
            cant_contains_others=True)
        print('==================之前的任务============')
        print('前99,get:23840   acc: 0.999748,total - skip=633 ')
        print('备注：try_test_acc_of_m1 Top3 23706,0.977567 (不互相包含23725,97.8351%) ')
        print('备注：TOP3 23706 0.993332 (不互相包含 前3,get:23706   acc: 0.994129 ')
    if True:
        # 这里回归一份新的extract_entitys_all_tj.resort_3.v4.8.txt,格式会变
        print('获取扩展的实体集合,并判断是否共有属性')
        bkh.expend_es_by_dict(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                              f2='../data/nlpcc2016/4-ner/extract_e/e1.dict.txt',
                              f3='../data/nlpcc2016/4-ner/result/q.rdf.score.v1.txt',
                              f4='../data/nlpcc2016/4-ner/result/q.rdf.score.expend.v1.txt',
                              f5='../data/nlpcc2016/4-ner/result/expend.v1.txt',
                              record=True,
                              compare=False)

    # if False:
    #     # 合并 q.rdf.txt.math_s.txt ， q.rdf 到 q.rdf.m_s
    #     bkh.rewrite_rdf(f3='../data/nlpcc2016/3-questions/q.rdf.txt',
    #                     f2='../data/nlpcc2016/3-questions/q.rdf.m_s.txt',
    #                     f1='../data/nlpcc2016/3-questions/q.rdf.txt.math_s.txt')
    #     print('重写q.rdf.m_s.txt')
    # if False:
    #     # 重新选择一遍属性
    #     bkh.choose_property(f1='../data/nlpcc2016/3-questions/q.rdf.m_s.txt',
    #                         f2='../data/nlpcc2016/3-questions/q.rdf.m_s.suggest.txt')
    # 重写rdf_extract_property_origin
    # C1.2.1
    if False:
        filter_list2 = cf.extract_property(f3='../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt',
                            f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                            f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin.txt',
                            skip=0,
                            skip_cant_match=True)
        print('重写q.rdf.ms.re.v1.filter.txt和rdf_extract_property_origin.txt')
        s1 = set(filter_list1) - set(filter_list2)
        print("%s "%' '.join(s1))

        s2 = set(filter_list2) - set(filter_list1)
        print("%s "%' '.join(s2))
    if False:
        # 仅用于测试
        cf.extract_property(f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                            f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter_test.txt',
                            f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin_test.txt',
                            skip=14610,
                            skip_cant_match=True)

    # 根据答案抽取出精简的KB
    if False:
        # F0.1.3
        # bkh.init_spo(config.cc_par('kb')) # 加载全部
        bkh.extract_kb_possible(f1='../data/nlpcc2016/2-kb/kb.v1.txt',
                                f2="../data/nlpcc2016/2-kb/kb-use.v1[20180408].txt",
                                f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                                f4='../data/nlpcc2016/4-ner/extract_entitys_all_tj.resort_3.expend.v1.txt')
        print('根据答案抽取出精简的KB kb-use.v2.txt')
    #     怪物(2014年李民基主演韩国电影)
    # 符号(2009年松本人志执导日本电影)
    # 郑成功(1987年香港tvb电视剧)

    if False:
        bkh.clean_baike_kb_repeat(f1="../data/nlpcc2016/2-kb/kb-use.v2.txt",
                                  f2="../data/nlpcc2016/2-kb/kb-use.v3.txt")
        print('替换指定属性')
    # 重写q.txt        # 3 生成新的训练文件
    if False:
        dh = DataClass(mode="cc", run_type='init')
        dh.build_all_q_r_tuple(99999999999999,
                               99999999999999, is_record=True)
        print('重新生成训练文件q_neg_r_tuple.v1')
    # 重生成所有测试集的候选属性
    if False:
        #  读取问题
        cf.build_test_ps(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                         f2='../data/nlpcc2016/5-class/test_ps.v4.txt', skip=14610)
    if False:
        cf.build_competing_ps(f1='../data/nlpcc2016/5-class/test_ps.v4.txt',
                              f2='../data/nlpcc2016/5-class/competing_ps.v1.txt')
