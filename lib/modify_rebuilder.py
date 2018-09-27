from lib.baike_helper import baike_helper, baike_test
from lib.classification_helper import classification
from lib.data_helper import DataClass
from lib.config import config
from lib.ct import ct
from lib.pretreatment import pretreatment

# 一键纠错
if __name__ == '__main__':  #
    bkt = baike_test()
    bkh = baike_helper(config.cc_par('alias_dict'))
    cf = classification()

    # -------------- 预处理 test 部分
    if False:
        pretreatment.re_write(f1='../data/nlpcc2016/10-test/nlpcc2018.kbqa.test',
                              f2='../data/nlpcc2016/10-test/test.txt')
    if False:
        pretreatment.stat_all_space(f1='../data/nlpcc2016/10-test/test.txt')
    # 重写别名指代
    if False:
        pretreatment.re_write_m2id(f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.kb.mention2id',
                                   f_out='../data/nlpcc2016/4-ner/extract_e/e1.dict.v2.txt')
    # 将实体
    if False:
        bkh.statistics_subject_len(f_in='../data/nlpcc2016/4-ner/extract_e/e1.dict.v2.txt',
                                   f_out='../data/nlpcc2016/4-ner/extract_e/e1.tj.v2-2.txt')

    # 答案选择
    if True:
        dh = DataClass("cc")
        s=r'F:\PycharmProjects\dl2\deeplearning\data\nlpcc2016\6-answer\select_100p\s.txt'
        p = r'F:\PycharmProjects\dl2\deeplearning\data\nlpcc2016\6-answer\select_100p\p.txt'
        dh.answer_select(f1_s=s,f2_r=p)
        # 19:59:03:	right:13516 wrong:582  valid	acc:0.958718	p1:0.816383	r1:0.958718	f1:0.848968
        # 19:59:03:	right:8587 wrong:708 	test	acc:0.923830	p1:0.825254	r1:0.923830	f1:0.848680
        # 100P
        # 17:49:22:	right:923 wrong:33  	valid	acc:0.965481	p1:0.618595	r1:0.965481	f1:0.674224
        # 17:49:22:	right:575 wrong:77  	test	acc:0.881902	 p1:0.626368	r1:0.881902	f1:0.669615

        #
    # -------------- NER识别 （从句子中生成NER）

    if False:
        bkt.n_gram_math_all(f_in="../data/nlpcc2016/4-ner/debug/q.rdf.ms.re.v1.txt",
                        f_out='../data/nlpcc2016/4-ner/debug/extract_entitys_all.txt',
                        f3="../data/nlpcc2016/4-ner/debug/e1.tj.v2.txt", # 实体别名的长度统计的文档
                        skip_no_space=False)
    if False:
        bkt.n_gram_math_all(f_in="../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt",
                        f_out='../data/nlpcc2016/4-ner/extract_entitys_all.v2.txt',
                        f3="../data/nlpcc2016/4-ner/extract_e/e1.tj.v2.txt", # 实体别名的长度统计的文档
                        skip_no_space=False)
    if False:
        # 根据extract_entitys_all.txt 重写e1.tj.v2，仅保留出现在候选实体中的
        bkt.rewrite_e1_tj(f1='../data/nlpcc2016/4-ner/extract_e/e1.dict.v2.txt',
                          f2='../data/nlpcc2016/4-ner/extract_entitys_all.txt',
                          f3='../data/nlpcc2016/4-ner/extract_e/e1.tj.simple.txt')
    if False:  # 统计完重写输出 排序
        bkt.file_tj(f1='../data/nlpcc2016/4-ner/extract_entitys_all.txt',  # 原始
                    f_out='../data/nlpcc2016/4-ner/extract_entitys_all_sort.txt',
                    record=True)

    # ------------- 将NER的结果相关的KB全部抽取出来
    if False:
        # 补全字典 ，根据漏掉的key从基本的字典中补全
        bkh.extract_kb_test2(f1='../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt',
                             f2='../data/nlpcc2016/4-ner/extract_e/e1.dict.txt',
                             f3='../data/nlpcc2016/2-kb/kb.v1.txt',
                             f_out='../data/nlpcc2016/10-test/kb-bu-4.txt',
                             )

    if False:
        bkh.extract_kb_test( f1='../data/nlpcc2016/10-test/test_extract_entitys.txt',
                        f2='../data/nlpcc2016/4-ner/extract_e/e1.tj.txt',
                        f3='../data/nlpcc2016/2-kb/kb.v1.txt',
                        f4='../data/nlpcc2016/10-test/kb-test.txt'
                        )

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

    if False:  #  skip=1086
        time_str = ct.time_path()
        num = 14
        filter_list1 = bkt.try_test_acc_of_m1(
            #   q.rdf.ms.re.v1.txt
            f1='../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt',
            f3='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v1.txt',
            # extract_entitys_v3                extract_entitys_all
            use_cx=False, use_expect=False, acc_index=[num],
            get_math_subject=True,
            f6='../data/nlpcc2016/4-ner/extract_entitys_all.txt.statistics.txt',
            f2='../data/nlpcc2016/4-ner/demo_20180904/q.rdf.txt.failed_v4.8_%d_%s.txt' % (num, time_str),
            f8='../data/nlpcc2016/4-ner/demo_20180904/extract_entitys_all_tj.resort_%d_%s.v4.8.txt' % (num, time_str),
            f9='../data/nlpcc2016/4-ner/demo_20180904/q.rdf.ms.re.top_%d_%s.v4.10.txt' % (num, time_str),
            f10='../data/nlpcc2016/4-ner/demo_20180904/ner_%d_%s.v4.10.txt' % (num, time_str),
            f13='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v4.txt',
            f12='../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt',
            combine_idf=False,
            cant_contains_others=False,
            test_top_1000=True,
            is_test =False)
    if False: # skip 669
        time_str = ct.time_path()
        num = 1
        filter_list1 = bkt.try_test_acc_of_s(
                #   q.rdf.ms.re.v1.txt
                f1='../data/nlpcc2016/6-answer/q.rdf.ms.re.v1.txt',
                f3='../data/nlpcc2016/4-ner/20180924_M2ID/extract_entitys_all.v1.txt',
                f4='../data/nlpcc2016/4-ner/extract_e/e1.dict.v2.txt',
                acc_index=[num],
                f6='../data/nlpcc2016/4-ner/extract_entitys_all.txt.statistics.txt',
                f2='../data/nlpcc2016/4-ner/demo_20180904/q.rdf.txt.failed_v4.8_%d_%s.txt' % (num, time_str),
                f8='../data/nlpcc2016/4-ner/demo_20180904/extract_entitys_all_tj.resort_%d_%s.v4.8.txt' % (
                num, time_str),
                f9='../data/nlpcc2016/4-ner/demo_20180904/q.rdf.ms.re.top_%d_%s.v4.10.txt' % (num, time_str),
                f10='../data/nlpcc2016/4-ner/demo_20180904/ner_%d_%s.v4.10.txt' % (num, time_str),
                f13='../data/nlpcc2016/4-ner/20180924_M2ID/extract_entitys_all_tj.v4.txt',
                f12='../data/nlpcc2016/4-ner/20180924_M2ID/q.rdf.ms.re.v4.txt',
                combine_idf=False,
                cant_contains_others=False,
                test_top_1000=True,
                is_test=False)
        #print('==================之前的任务============')
        #print('前99,get:23840   acc: 0.999748,total - skip=633 ')
        #print('备注：try_test_acc_of_m1 Top3 23706,0.977567 (不互相包含23725,97.8351%) ')
        #print('备注：TOP3 23706 0.993332 (不互相包含 前3,get:23706   acc: 0.994129 ')
    # 这里需要LR_TRAIN一次
    if False:
        # 这里回归一份新的extract_entitys_all_tj.resort_3.v4.8.txt,格式会变
        print('获取扩展的实体集合,并判断是否共有属性')
        bkh.expend_es_by_dict(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                              f2='../data/nlpcc2016/4-ner/extract_e/e1.dict.txt',
                              f3='../data/nlpcc2016/4-ner/demo3/q.rdf.score.top_3_all_0.v4.10.txt',
                              f4='../data/nlpcc2016/4-ner/result/q.rdf.score.expend.v1.txt',  # 输出
                              f5='../data/nlpcc2016/4-ner/result/expend.v1.txt',  # 输出
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

    # --------------  问答
    if False:
        # 弃用
        filter_list2 = cf.extract_property(f3='../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt',
                                           f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v2.filter.txt',
                                           f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin.txt',
                                           skip=0,
                                           skip_cant_match=True)
        print('重写q.rdf.ms.re.v1.filter.txt')
    if False:
        # "rdf_extract_property_origin.txt"
        filter_list3 = cf.extract_property2(f1='../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt',
                                           f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin.v2.txt',
                                           )
        pass
        # s1 = set(filter_list1) - set(filter_list2)
        # print("filter_list1-filter_list2: %s "%' '.join(s1))
        # 作者	30	0	75	212	1397	1511	1692	2969
        # s2 = set(filter_list2) - set(filter_list1)
        # print("%s "%' '.join(s2))
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
                                f2="../data/nlpcc2016/2-kb/kb-use.v1[20180908].txt",
                                f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                                f4='../data/nlpcc2016/4-ner/extract_entitys_all_tj.v2.txt')
        print('根据答案抽取出精简的KB kb-use.v2.txt')
    # 怪物(2014年李民基主演韩国电影)
    # 符号(2009年松本人志执导日本电影)
    # 郑成功(1987年香港tvb电视剧)

    if False:
        bkh.clean_baike_kb_repeat(f1="../data/nlpcc2016/2-kb/kb-use.v2.txt",
                                  f2="../data/nlpcc2016/2-kb/kb-use.v3.txt")
        print('替换指定属性 后缀1 2 3 4 ')
    # 重写q.txt        # 3 生成新的训练文件
    if False:
        dh = DataClass(mode="cc", run_type='init')
        dh.build_all_q_r_tuple(99999999999999,
                               99999999999999, is_record=True)
        print('重新生成训练文件q_neg_r_tuple.v1')

    # ============================================================
    # 竞争属性部分
    # ============================================================
    # 重生成所有测试集的候选属性
    # 1 构造KB中的可能的竞争子集合
    if False:
        kb_path = config.cc_par('kb')
        cf.build_competing_p_in_kb('../data/nlpcc2016/13-competing/competing_p_in_kb.v2.txt',
                                   '../data/nlpcc2016/13-competing/competing_s_in_kb.v2.txt',
                                   kb_path)
    if False:
        #  读取问题
        f1 = config.cc_par('cc_q_path')
        skip = config.cc_par('real_split_train_test_skip_v2')
        cf.build_test_ps(f1=f1,
                         f2='../data/nlpcc2016/13-competing/train_ps.v1.txt', skip=skip)

    if False:
        cf.build_competing_ps(f1='../data/nlpcc2016/13-competing/train_ps.v1.txt',
                              f2='../data/nlpcc2016/13-competing/competing_ps.v1.txt',
                              f3='../data/nlpcc2016/13-competing/competing_ps_tj.v2.txt')

    # 检查所有的实体是否存在KV中
    #
    if False:
        dh = DataClass("cc")
        dh.check_spo()





