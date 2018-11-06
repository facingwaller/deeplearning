# 方法3
# def ns_competing_v3(dh, discriminator,  sess, step):
#     ct.print('ns_epoches V3 start')
#     ns_r_r_score_all = []
#
#     r_cp = []
#     ns_index = 0
#     cp_dict = dict()
#
#     # 更新 全部属性
#     gc1 = dh.batch_iter_all_competing_ps()
#     for item in gc1:
#         ns_index += 1
#         state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
#         feed_dict = {}
#         feed_dict[discriminator.ns_r_cp] = item['r_cp']
#         ns_r_r_state, _ = sess.run(
#             [discriminator.ns_test_r_cp_out,
#              discriminator.ns_r_cp],
#             feed_dict=feed_dict)
#
#         ns_r_r_score_all.extend(ns_r_r_state)
#         r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
#         # 更新每个属性隐藏层变量进字典
#         for i in range(0, len(ns_r_r_state)):
#             score = ns_r_r_state[i]
#             r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
#             cp_dict[r_cp_str] = score
#
#     # 更新 全部问题
#     ns_index = 0
#     ns_q_state_all = []
#     gc1 = dh.batch_iter_all_questions()
#     for item in gc1:
#         ns_index += 1
#         feed_dict = {}
#         feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
#         [ns_q_state, _] = sess.run(
#             [discriminator.ns_test_q_out,
#              discriminator.ns_q],
#             feed_dict=feed_dict)
#         ns_q_state_all.extend(ns_q_state)
#         # 遍历得分
#
#     # 更新 全部实体
#     ns_index = 0
#     ns_s_state_all = []
#     q_s_all = []
#     ns_s_state_all_dict = {}
#     gc1 = dh.batch_iter_all_entitys()
#     for item in gc1:
#         ns_index += 1
#         feed_dict = {}
#         q_s = item['s_cand']  #  用于识别s的候选s
#         q_s_all.extend(q_s)
#         feed_dict[discriminator.ns_q] = q_s  # negative sampling
#         [ns_s_state, _] = sess.run(
#             [discriminator.ner_test_input_r,
#              discriminator.ner_test_r],
#             feed_dict=feed_dict)
#         ns_s_state_all.extend(ns_s_state)
#         # 改成根据KEY得到实体向量
#         for _,_state in zip(q_s,ns_s_state_all):
#             _key = dh.converter.arr_to_text_no_unk(_)
#             ns_s_state_all_dict[_key] = _state
#
#
#     # 遍历问题,计算q-r
#     dh.question_comcpeting_ps = []
#     ct.print("@==================" , 'update_score')
#     time_start = time.time()
#     for global_index in range(len(ns_q_state_all)):
#         if global_index % 100 == 0:
#             print('cost %d/%d: %s'% (global_index ,len(ns_q_state_all), time.time() - time_start))
#             time_start = time.time()
#         # 获取问题的竞争属性
#         r_pos1 = dh.relation_list[global_index]  # 正确的属性
#         p_v = dh.competing_train_dict.get(r_pos1, '')
#         p_v = [x[0] for x in p_v]  # 去除频率
#         p_v.insert(0,r_pos1)  # 加入自己
#         p_v_state = []
#         for x in p_v:
#             _es = cp_dict.get(x,'')
#             if _es != '':
#                 p_v_state.append(_es)     # 取出state
#             else:
#                 ct.print(x, 'ns_competing_v2')
#
#         temp_ns_q_state_list = [ns_q_state_all[global_index] for x
#                                 in range(len(p_v_state))]
#         feed_dict = {}
#         feed_dict[discriminator.ns2_q] = p_v_state  # negative sampling 问题
#         feed_dict[discriminator.ns2_r] = temp_ns_q_state_list  # negative sampling 属性
#
#         try:
#             [ns2_q_r_score] = sess.run(
#                 [discriminator.ns2_q_r],
#                 feed_dict=feed_dict)
#         except Exception as e1:
#             print(e1)
#
#
#         st_list = []
#         _competing_train_dict_set = set()
#         for _index in range(len(ns2_q_r_score)):
#             st = ct.new_struct()
#             st.index = _index
#             st.p = p_v[_index]
#             st.label = st.p == r_pos1
#             st.score = max(0,ns2_q_r_score[_index]) # 保持非负数
#             # if st.score < 0 or st.score == None:
#             #     ct.print('st.score<0 ','bug')
#             st_list.append(st)
#             _tp = (st.p, st.score)  # 属性 ，得分
#             _competing_train_dict_set.add(_tp)
#
#         # 加入
#         _r_pos1_score = st_list[0].score # 第一个是正确属性
#         del st_list[0] # 删除pos , 偶尔同p计算的score不一样 考虑是否增加一个margin
#         st_list = list(filter(lambda x: x.score > _r_pos1_score, st_list))
#         # _r_pos1_score - config.cc_par('loss_margin')
#         st_list.sort(key=ct.get_key)
#         st_list.reverse()
#
#         _msg = []
#         q_k = dh.question_list_origin[global_index]  # 问题
#         for item in st_list:
#             _msg.append("%s_%s" % (item.p, str(item.score)))
#         is_train = dh.question_labels[global_index]
#         ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
#                  (str(is_train),q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
#                  'update_score')
#         # 记录更新 K-V
#         num = len(st_list) # 个数
#         _v1 = dh.competing_train_p_id_num.get(r_pos1, '')
#         _t1 = (global_index, num)
#         if _v1 != '':
#             if _v1[1] >= num: # 已经存在的少于当前的
#                 _t1 = _v1
#         dh.competing_train_p_id_num[r_pos1] = _t1
#         dh.question_comcpeting_ps.append(_competing_train_dict_set)
#
#     # 遍历问题，计算q-s
#     ct.print("@==================", 'update_ner_score_ner')
#     for global_index in range(len(dh.question_list_origin)):
#         if dh.question_labels[global_index]: # 只做训练集，测试集就停下来
#             break
#         if global_index % 100 == 0:
#             print('cost %d/%d: %s'% (global_index ,len(ns_q_state_all), time.time() - time_start))
#             time_start = time.time()
#
#         # 获取实体的竞争实体
#         #
# 方法2

# 改为只加入超过的
def ns_competing_ert(dh, discriminator,  sess, step):
    ct.print('ns_epoches  ns_competing_ert start')
    ns_r_r_score_all = []

    r_cp = []
    ns_index = 0
    cp_dict = dict()

    # 更新 全部属性
    gc1 = dh.batch_iter_all_competing_ps()
    for item in gc1:
        ns_index += 1
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_r_cp] = item['r_cp']
        ns_r_r_state, _ = sess.run(
            [discriminator.ns_test_r_cp_out,
             discriminator.ns_r_cp],
            feed_dict=feed_dict)

        ns_r_r_score_all.extend(ns_r_r_state)
        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
        # 更新每个属性隐藏层变量进字典
        for i in range(0, len(ns_r_r_state)):
            score = ns_r_r_state[i]
            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
            cp_dict[r_cp_str] = score

    # 更新 全部问题
    ns_index = 0
    ns_q_state_all = []
    gc1 = dh.batch_iter_all_questions()
    for item in gc1:
        ns_index += 1
        feed_dict = {}
        feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
        [ns_q_state, _] = sess.run(
            [discriminator.ns_test_q_out,
             discriminator.ns_q],
            feed_dict=feed_dict)
        ns_q_state_all.extend(ns_q_state)
        # 遍历得分

    # 遍历问题
    dh.question_comcpeting_ps = []
    ct.print("@==================ns_competing_ert" , 'update_score')
    time_start = time.time()
    for global_index in range(len(ns_q_state_all)): # 遍历所有问题
        if global_index % 100 == 0:
            print('cost %d/%d: %s'% (global_index ,len(ns_q_state_all), time.time() - time_start))
            time_start = time.time()
        # 获取问题的竞争属性
        r_pos1 = dh.relation_list[global_index]  # 正确的属性
        p_v = dh.competing_train_dict.get(r_pos1, '') # 获取P的竞争属性集合
        if p_v == '':
            ct.print('%s'%r_pos1,'bug_competing_train_dict') # 没找到与其竞争的,使用默认的

        p_v = [x[0] for x in p_v]  # 去除频率
        p_v.insert(0,r_pos1)  # 加入自己
        p_v_state = []
        _pv = []
        for x in p_v:
            _es = cp_dict.get(x,'')  # 获取该属性的state 隐藏层向量
            if _es != '':
                p_v_state.append(_es)     # 取出state
                _pv.append(x)
            else:
                ct.print("%s - %s"%(r_pos1,x), 'ns_competing_v2') #  片长英文名
        # continue
        p_v = _pv # 去掉找不到的
        temp_ns_q_state_list = [ns_q_state_all[global_index] for x
                                in range(len(p_v_state))]
        feed_dict = {}
        feed_dict[discriminator.ns2_q] = p_v_state  # negative sampling 问题
        feed_dict[discriminator.ns2_r] = temp_ns_q_state_list  # negative sampling 属性

        # try:
        [ns2_q_r_score] = sess.run(
                [discriminator.ns2_q_r],
                feed_dict=feed_dict)
        # except Exception as e1:
        #     print(e1)

        st_list = []
        _competing_train_dict_set = set()
        for _index in range(len(ns2_q_r_score)):

            st = ct.new_struct()
            try:
                st.index = _index
                st.p = p_v[_index]
                st.label = st.p == r_pos1
                st.score = max(0,ns2_q_r_score[_index])  # 保持非负数
            except Exception as e1:
                print(e1)
            # if st.score < 0 or st.score == None:
            #     ct.print('st.score<0 ','bug')
            st_list.append(st)
            # _tp = (st.p, st.score)  # 属性 ，得分
            # _competing_train_dict_set.add(_tp)

        # 加入
        _r_pos1_score = st_list[0].score # 第一个是正确属性
        del st_list[0] # 删除pos , 偶尔同p计算的score不一样 考虑是否增加一个margin
        st_list = list(filter(lambda x: x.score > _r_pos1_score, st_list))
        # _r_pos1_score - config.cc_par('loss_margin')
        st_list.sort(key=ct.get_key)
        st_list.reverse()

        _msg = []
        q_k = dh.question_list_origin[global_index]  # 问题
        for item in st_list:
            _msg.append("%s_%s" % (item.p, str(item.score)))
            _tp = (item.p, item.score)  # 属性 ，得分
            _competing_train_dict_set.add(_tp)
        is_train = dh.question_labels[global_index]
        ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                 (str(is_train),q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                 'update_score')
        # 记录更新 K-V
        num = len(st_list) # 个数
        _v1 = dh.competing_train_p_id_num.get(r_pos1, '')
        _t1 = (global_index, num)
        if _v1 != '':
            if _v1[1] >= num: # 已经存在的少于当前的
                _t1 = _v1
        dh.competing_train_p_id_num[r_pos1] = _t1
        dh.question_comcpeting_ps.append(_competing_train_dict_set)

def ns_competing_v2(dh, discriminator,  sess, step):
    ct.print('ns_epoches start')
    ns_r_r_score_all = []

    r_cp = []
    ns_index = 0
    cp_dict = dict()

    # 更新 全部属性
    gc1 = dh.batch_iter_all_competing_ps()
    for item in gc1:
        ns_index += 1
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_r_cp] = item['r_cp']
        ns_r_r_state, _ = sess.run(
            [discriminator.ns_test_r_cp_out,
             discriminator.ns_r_cp],
            feed_dict=feed_dict)

        ns_r_r_score_all.extend(ns_r_r_state)
        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
        # 更新每个属性隐藏层变量进字典
        for i in range(0, len(ns_r_r_state)):
            score = ns_r_r_state[i]
            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
            cp_dict[r_cp_str] = score

    # 更新 全部问题
    ns_index = 0
    ns_q_state_all = []
    gc1 = dh.batch_iter_all_questions()
    for item in gc1:
        ns_index += 1
        feed_dict = {}
        feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
        [ns_q_state, _] = sess.run(
            [discriminator.ns_test_q_out,
             discriminator.ns_q],
            feed_dict=feed_dict)
        ns_q_state_all.extend(ns_q_state)
        # 遍历得分

    # 遍历问题
    dh.question_comcpeting_ps = []
    ct.print("@==================ns_competing_v2" , 'update_score')
    time_start = time.time()
    for global_index in range(len(ns_q_state_all)): # 遍历所有问题
        if global_index % 100 == 0:
            print('cost %d/%d: %s'% (global_index ,len(ns_q_state_all), time.time() - time_start))
            time_start = time.time()
        # 获取问题的竞争属性
        r_pos1 = dh.relation_list[global_index]  # 正确的属性
        p_v = dh.competing_train_dict.get(r_pos1, '') # 获取P的竞争属性集合
        if p_v == '':
            ct.print('%s '%r_pos1,'bug')

        p_v = [x[0] for x in p_v]  # 去除频率
        p_v.insert(0,r_pos1)  # 加入自己
        p_v_state = []
        _pv = []
        for x in p_v:
            _es = cp_dict.get(x,'')  # 获取该属性的state 隐藏层向量
            if _es != '':
                p_v_state.append(_es)     # 取出state
                _pv.append(x)
            else:
                ct.print("%s - %s"%(r_pos1,x), 'ns_competing_v2') #  片长英文名
        # continue
        p_v = _pv # 去掉找不到的
        temp_ns_q_state_list = [ns_q_state_all[global_index] for x
                                in range(len(p_v_state))]
        feed_dict = {}
        feed_dict[discriminator.ns2_q] = p_v_state  # negative sampling 问题
        feed_dict[discriminator.ns2_r] = temp_ns_q_state_list  # negative sampling 属性

        # try:
        [ns2_q_r_score] = sess.run(
                [discriminator.ns2_q_r],
                feed_dict=feed_dict)
        # except Exception as e1:
        #     print(e1)

        st_list = []
        _competing_train_dict_set = set()
        for _index,_label in zip(range(len(ns2_q_r_score)),dh.question_labels):

            st = ct.new_struct()
            try:
                st.index = _index
                st.p = p_v[_index]
                st.label = st.p == r_pos1
                st.score = max(0,ns2_q_r_score[_index])  # 保持非负数
            except Exception as e1:
                print(e1)
            # if st.score < 0 or st.score == None:
            #     ct.print('st.score<0 ','bug')
            st_list.append(st)
            _tp = (st.p, st.score)  # 属性 ，得分
            _competing_train_dict_set.add(_tp)

        # 加入
        _r_pos1_score = st_list[0].score # 第一个是正确属性
        del st_list[0] # 删除pos , 偶尔同p计算的score不一样 考虑是否增加一个margin
        st_list = list(filter(lambda x: x.score > _r_pos1_score, st_list))
        # _r_pos1_score - config.cc_par('loss_margin')
        st_list.sort(key=ct.get_key)
        st_list.reverse()

        _msg = []
        q_k = dh.question_list_origin[global_index]  # 问题
        for item in st_list:
            _msg.append("%s_%s" % (item.p, str(item.score)))
        is_train = dh.question_labels[global_index]
        ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                 (str(is_train),q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                 'update_score')

        # 记录更新 K-V
        num = len(st_list) # 个数
        _v1 = dh.competing_train_p_id_num.get(r_pos1, '')
        _t1 = (global_index, num)
        if _v1 != '':
            if _v1[1] >= num: # 已经存在的少于当前的
                _t1 = _v1
        dh.competing_train_p_id_num[r_pos1] = _t1
        dh.question_comcpeting_ps.append(_competing_train_dict_set)

# 是否加入全部
# 是否只加入训练
def ns_competing_relation(dh, discriminator,  sess, step,is_add_all = False):
    ct.print('ns_epoches start')
    ns_r_r_score_all = []

    r_cp = []
    ns_index = 0
    cp_dict = dict()

    # 更新 全部属性
    gc1 = dh.batch_iter_all_competing_ps()
    for item in gc1:
        ns_index += 1
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_r_cp] = item['r_cp']
        ns_r_r_state, _ = sess.run(
            [discriminator.ns_test_r_cp_out,
             discriminator.ns_r_cp],
            feed_dict=feed_dict)

        ns_r_r_score_all.extend(ns_r_r_state)
        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
        # 更新每个属性隐藏层变量进字典
        for i in range(0, len(ns_r_r_state)):
            score = ns_r_r_state[i]
            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
            cp_dict[r_cp_str] = score

    # 更新 全部问题
    ns_index = 0
    ns_q_state_all = []
    ns_q_str = []
    gc1 = dh.batch_iter_all_questions()
    for item in gc1:
        ns_index += 1
        feed_dict = {}
        feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
        [ns_q_state, _] = sess.run(
            [discriminator.ns_test_q_out,
             discriminator.ns_q],
            feed_dict=feed_dict)
        ns_q_state_all.extend(ns_q_state)
        # 属性的对应字符串
        _1 = [dh.converter.arr_to_text_no_unk(x)   for x  in  item['q_p']]
        ns_q_str.extend(_1)
        # 遍历得分

    # 遍历问题
    dh.question_comcpeting_ps = []
    dh.competing_train_p_id_num.clear()
    ct.print("@==================ns_competing_v2" , 'update_score')
    time_start = time.time()
    for global_index in range(len(ns_q_state_all)): # 遍历所有问题
        if global_index % 100 == 0:
            print('cost %d/%d: %s'% (global_index ,len(ns_q_state_all), time.time() - time_start))
            time_start = time.time()
        # 获取问题的竞争属性
        r_pos1 = dh.relation_list[global_index]  # 正确的属性
        q_origin = dh.question_list_origin[global_index]
        p_v = dh.competing_train_dict.get(r_pos1, '') # 获取P的竞争属性集合
        if p_v == '':
            ct.print('%s '%r_pos1,'bug')

        p_v = [x[0] for x in p_v]  # 去除频率

        # 尝试 不包含P_POS
        # if config.cc_par('only_p_neg_in_cp'):
        #     p_v = list(set(p_v) -  set(dh.relation_list))
        # else:
        #     pass

        p_v.insert(0,r_pos1)  # 加入自己
        p_v_state = []
        _pv = []
        if config.cc_par('convert_rs_to_words'):
            p_v = ct.convert_rs_to_words(p_v)
        else:
            pass


        for x in p_v:
            _es = cp_dict.get(x,'')  # 获取该属性的state 隐藏层向量
            if _es != '':
                p_v_state.append(_es)     # 取出state
                _pv.append(x)
            else:
                ct.print("%s - %s"%(r_pos1,x), 'ns_competing_v2_not_exist') #  片长英文名
        # continue
        p_v = _pv # 去掉找不到的
        temp_ns_q_state_list = [ns_q_state_all[global_index] for x
                                in range(len(p_v_state))]
        feed_dict = {}
        feed_dict[discriminator.ns2_q] = p_v_state  # negative sampling 问题
        feed_dict[discriminator.ns2_r] = temp_ns_q_state_list  # negative sampling 属性

        # 记录

        for x in p_v:
            ct.print('%s\t%s'%(ns_q_str[global_index],x),'ns_q_p')
            # pass

        # try:
        [ns2_q_r_score] = sess.run(
                [discriminator.ns2_q_r],
                feed_dict=feed_dict)
        # except Exception as e1:
        #     print(e1)

        st_list = []
        _competing_train_dict_set = set()
        for _index in range(len(ns2_q_r_score)):
            st = ct.new_struct()
            try:
                st.index = _index
                st.p = p_v[_index]
                st.label = st.p == r_pos1
                st.score = max(0,ns2_q_r_score[_index])  # 保持非负数
            except Exception as e1:
                print(e1)
            # if st.score < 0 or st.score == None:
            #     ct.print('st.score<0 ','bug')
            st_list.append(st)
            # _tp = (st.p, st.score)  # 属性 ，得分
            # _competing_train_dict_set.add(_tp)

        # 加入
        _r_pos1_score = st_list[0].score # 第一个是正确属性
        if True:
            del st_list[0] # 删除pos , 偶尔同p计算的score不一样 考虑是否增加一个margin
            padding = 0.001 # 作为一个间隔
            loss_margin  = config.cc_par('loss_margin')
            st_list = list(filter(lambda x: x.score > (_r_pos1_score + padding), st_list))
            # _r_pos1_score - config.cc_par('loss_margin')
            # _r_pos1_score
            st_list.sort(key=ct.get_key)
            # st_list.reverse()
            # 临时测试：过滤掉包含原属性的,
            st_list = list(filter(lambda x: not str(x.p).__contains__(r_pos1), st_list))
        else:
            st_list.sort(key=ct.get_key)


        # 过滤掉出现在句子中的（可能更符合），
        # st_list = list(filter(lambda x: not str(q_origin).__contains__(x.p), st_list))
        # 临时测试改为取5个分数最低的
        # 超过5个才取
        # if len(st_list)>5:
        #     st_list = st_list[0:5]
        # else:
        #     st_list = []

        _msg = []
        q_k = dh.question_list_origin[global_index]  # 问题
        is_test = dh.question_labels[global_index]  # 是训练集还是测试集
        for item in st_list:
            _msg.append("%s_%s" % (item.p, str(item.score)))
            _tp = (item.p, item.score)  # 属性 ，得分
            if not is_test:  # 只加入训练的
                _competing_train_dict_set.add(_tp)

        ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                 (str(is_test),q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                 'update_score')
        # 记录更新 K-V

        num = len(st_list)  # 个数
        _v1 = dh.competing_train_p_id_num.get(r_pos1, '')
        _t1 = (global_index, num) # dh.question_global_index[global_index]
        if _v1 != '':
            if _v1[1] >= num: # 已经存在的少于当前的
                _t1 = _v1 # 将最好的赋值到当前的，保持之前最好的
                if is_test:  # 打印测试集的
                    ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                             (str(is_test), q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                             'update_score_test')
        if not is_test:  # 只加入训练集的
            dh.competing_train_p_id_num[r_pos1] = _t1 # 更新该属性最弱的问题
        dh.question_comcpeting_ps.append(_competing_train_dict_set)

def ns_competing_v1(dh, discriminator,  sess, step):
    ct.print('ns_epoches start')
    ns_r_r_score_all = []
    ns_q_state_all = []
    r_cp = []
    ns_index = 0
    cp_dict = dict()
    # 更新 全部属性
    gc1 = dh.batch_iter_all_competing_ps()
    for item in gc1:
        ns_index += 1
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_r_cp] = item['r_cp']
        ns_r_r_state, _ = sess.run(
            [discriminator.ns_test_r_cp_out,
             discriminator.ns_r_cp],
            feed_dict=feed_dict)

        if ns_index % 100 == 0:
            print(state)
        ns_r_r_score_all.extend(ns_r_r_state)
        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
        # 更新每个属性隐藏层变量进字典
        for i in range(0, len(ns_r_r_state)):
            score = ns_r_r_state[i]
            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
            cp_dict[r_cp_str] = score
    # 更新 全部问题
    ns_index = 0
    gc1 = dh.batch_iter_all_questions()
    for item in gc1:
        ns_index += 1
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
        [ns_q_state, _] = sess.run(
            [discriminator.ns_test_q_out,
             discriminator.ns_q],
            feed_dict=feed_dict)
        ns_q_state_all.extend(ns_q_state)
        # 遍历得分
    top_n = 20
    dh.update_competing_q_p_cosine(cp_dict, ns_q_state_all)
