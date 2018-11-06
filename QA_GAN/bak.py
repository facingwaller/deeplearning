def judging_quality(dh,tag,discriminator,  sess, train_neg_gan_k, train_pos_gan_k, train_q_gan_k):
    lose,win = 0,0
    feed_dict = {
        discriminator.ori_input_quests: train_q_gan_k,  # ori_batch
        discriminator.cand_input_quests: train_pos_gan_k,  # cand_batch
        discriminator.neg_input_quests: train_neg_gan_k  # neg_batch
    }
    # 给D计算出reward
    reward = sess.run(discriminator.reward,feed_dict)
    # reward= 2 * (tf.sigmoid( 0.05- (q_pos -q_neg) ) - 0.5)
    for _reward in reward:
        ct.print(_reward, 'reward')
    hasWin = False
    neg_str = ''
    for x,neg in zip(reward,train_neg_gan_k):
        if x < 0:
            win += 1
            neg_str = neg_str + '\t'+ dh.converter.arr_to_text_no_unk(neg)
        else:
            lose += 1
    hasWin = win> 0
    msg = "%s\t%s\t%d\t%d%s"%( dh.converter.arr_to_text_no_unk(train_q_gan_k[0]),str(hasWin),win,lose,neg_str)
    ct.print(msg,'jq_%s'%tag)
    return lose, reward, win

# g 根据候选生成最终的负例
def gen_neg(generator, sess, train_neg, train_pos, train_q):
    if config.cc_compare('pool_mode', 'additional') or \
            config.cc_compare('pool_mode', 'competing_ps') or \
            config.cc_compare('pool_mode', 'only_default'):
        # 2 随机取100个neg
        feed_dict = {
            generator.ori_input_quests: train_q,  # ori_batch
            generator.cand_input_quests: train_pos,  # cand_batch
            generator.neg_input_quests: train_neg  # neg_batch
        }

        # 生成预测 # cosine(q,neg) - cosine(q,pos) 正常应该是负数
        # 在QA中是排名cosine取最高的作为正确的。这里通过QA_CNN计算出Q_NEG - Q_POS的得分差值
        # predicteds = []
        predicteds = sess.run(generator.gan_score, feed_dict=feed_dict)
        exp_rating = np.exp(np.array(predicteds) * FLAGS.sampled_temperature)
        prob = exp_rating / np.sum(exp_rating)
        ct.check_inf(predicteds)

        pools = train_neg
        gan_k = FLAGS.gan_k  # + r_len / 限定个数
        if gan_k > len(pools):
            # raise ('从pool中取出的item数目不能超过从pool中item的总数')
            gan_k = len(pools)
            if config.cc_par('pool_mode') != 'only_default':
                ct.print('only_default 除非否则报错。FLAGS.gan_k > len(pools) %d ' % gan_k, 'error')
        elif gan_k < FLAGS.gan_k:
            gan_k = FLAGS.gan_k
        neg_index = np.random.choice(np.arange(len(pools)), size=gan_k, p=prob,
                                     replace=False)  # 生成 FLAGS.gan_k个负例
        # 根据neg index 重新选
        train_q_gan_k = []
        train_neg_gan_k = []
        train_pos_gan_l = []
        for i in neg_index:
            train_neg_gan_k.append(train_neg[i])
            train_q_gan_k.append(train_q[i])
            train_pos_gan_l.append(train_pos[i])
        train_q = train_q_gan_k
        train_pos = train_pos_gan_l
        train_neg = train_neg_gan_k
    else:
        raise Exception('NO ')

    return train_neg, train_pos, train_q
# --------------- G model
g_index = 0
_ns_model = 'only_default'  # competing_q  only_default random
for g_index in range(FLAGS.g_epoches):
    # model_list = ['random', 'competing_q']
    # for _ns_model in model_list:
    state = "step=%d_epoches=%s_index=%d" % (step, 'g', g_index)
    ct.print(state)
    # if False:
    toogle_line = "G model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
        step, len(dh.q_neg_r_tuple))
    ct.log3(toogle_line)
    ct.just_log2("info", toogle_line)

    train_part = 'relation'
    model = 'train'
    # 1 遍历raw
    shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
    win, lose = 0, 0
    for index in shuffle_indices:
        train_step += 1
        # 取出一个问题的相关数据
        # train_q, train_pos, train_neg, r_len = \
        #     dh.batch_iter_gan_train(dh.train_question_list_index,
        #                             dh.train_relation_list_index, model,
        #                             index, train_part,
        #                             FLAGS.batch_size_gan,
        #                             config.cc_par('pool_mode'))
        # _ns_model = 'only_default'
        batch_size = FLAGS.batch_size

        data_dict = dh.batch_iter_cand_s_p_not_yield(model, index, batch_size, _ns_model)
        train_q, train_pos, train_neg = \
            data_dict['q_p'], data_dict['p_pos'], data_dict['p_neg']
        if len(train_q) == 0:
            continue

        # 2 随机取100个neg
        feed_dict = {
            generator.ori_input_quests: train_q,  # ori_batch
            generator.cand_input_quests: train_pos,  # cand_batch
            generator.neg_input_quests: train_neg  # neg_batch
        }

        # 生成预测 # cosine(q,neg) - cosine(q,pos) 正常应该是负数
        # 在QA中是排名cosine取最高的作为正确的。这里通过QA_CNN计算出Q_NEG - Q_POS的得分差值
        # predicteds = []
        predicteds = sess.run(generator.gan_score, feed_dict)
        exp_rating = np.exp(np.array(predicteds) * FLAGS.sampled_temperature)
        prob = exp_rating / np.sum(exp_rating)
        #
        ct.check_inf(predicteds)
        # 遍历记录
        debug_gan2 = []
        for i in range(len(predicteds)):
            # debug_gan2.append("predicted_%d\t%s\t%s" %
            #               (i, dh.converter.arr_to_text_no_unk(train_neg[i]), prob[i]))
            ct.just_log2("info", "predicted_%d\t%s\t%s\t%s" %
                         (i, dh.converter.arr_to_text_no_unk(train_neg[i]), predicteds[i], prob[i]))
        # predicteds_list = [x for x in predicteds]
        # predicteds_list.sort()
        # rrr1 = '\t'.join([str(x) for x in predicteds_list])
        # ct.print("#%d#\t%s" % (index, rrr1), 'debug_predicteds_list')

        pools = train_neg
        gan_k = 5  # FLAGS.gan_k + r_len
        use_top_k = False
        if use_top_k:  # 直接取前X个
            ts = []
            for i in range(len(predicteds)):
                t1 = (i, predicteds[i])
                ts.append(t1)
            ts1 = sorted(ts, key=lambda x: x[1], reverse=True)
            ts1 = ts1[0:gan_k]
            neg_index = [x[0] for x in ts1]

        else:
            if FLAGS.gan_k > len(pools):
                # raise ('从pool中取出的item数目不能超过从pool中item的总数')
                gan_k = len(pools)
            try:
                neg_index = np.random.choice(np.arange(len(pools)), size=gan_k, p=prob,
                                             replace=False)  # 生成 FLAGS.gan_k个负例
            except Exception as e1:
                print(e1)
                raise (e1)
        # 根据neg index 重新选
        train_q_gan_k = []
        train_neg_gan_k = []
        train_pos_gan_k = []
        debug_gan1 = []
        for i in neg_index:
            train_neg_gan_k.append(train_neg[i])  # 记录下来
            train_q_gan_k.append(train_q[i])
            train_pos_gan_k.append(train_pos[i])
            # 前X的neg是  index 文本 D给的得分 ,prob 回归后的概率
            # ct.just_log2("info", )
            debug_gan1.append("top_%d\t%s\t%s" %
                              (i, dh.converter.arr_to_text_no_unk(train_neg[i]), prob[i]))
        # 取出这些负样本就拿去给D判别 score12 = q_pos   score13 = q_neg
        tag = 'test'
        lose, reward, win = judging_quality(dh, tag, discriminator, sess, train_neg_gan_k, train_pos_gan_k,
                                            train_q_gan_k)
        # if neg_better_than_pos:
        #     win += 1
        # else:
        #     lose += 1
        # reward_list = [x for x in reward]
        # reward_list.sort()
        # rrr1 = '\t'.join([str(x) for x in reward_list])
        # ct.just_log2("info", "reward_list\t%d\t%s" % (index, rrr1))
        for i in range(len(reward)):
            debug_gan1[i] += "\t%s" % reward[i]
            ct.just_log2("info", "%s" % (debug_gan1[i]))
        # 记录每个属性对应的奖励


        # 用reward训练G
        feed_dict = {
            generator.ori_input_quests: train_q,  # ori_batch
            generator.cand_input_quests: train_pos,  # cand_batch
            generator.neg_index: neg_index,
            generator.neg_input_quests: train_neg,  # neg_batch
            generator.reward: reward}
        # 原作者：应该是全集上的softmax	但是此处做全集的softmax开销太大了
        _, run_step, current_loss, positive, negative, \
        _1, _2 = sess.run(
            [generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,
             generator.negative,
             generator.prob, generator.reward],  # self.prob= tf.nn.softmax( self.cos_13)
            feed_dict)  # self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)
        _1_log = np.log(_1)
        _ganlos = -np.mean(_1_log * _2)
        line = ("%s-%s: G step %d, loss %f  positive %f negative %f" % (
            train_step, len(shuffle_indices), run_step, current_loss, positive, negative))
        loss_dict['loss'] += current_loss
        loss_dict['pos'] += positive
        loss_dict['neg'] += negative
        ct.print(line, 'loss')

    # check
    total = len(shuffle_indices)
    msg = "%s\tloss=%2.6f\tpos=%2.6f\tneg=%2.6f;G_win=%d lose = %d \tacc=%s" % (state, loss_dict['loss'] / total,
                                                                                loss_dict['pos'] / total,
                                                                                loss_dict['neg'] / total
                                                                                , lose, win, lose / (win + lose))
    ct.print(msg, 'debug_gan')
    loss_dict['loss'] = 0
    loss_dict['pos'] = 0
    loss_dict['neg'] = 0

    # 验证 和测试
    # elvation(state, 0, dh, step, sess, discriminator, merged, writer, valid_test_dict,
    #          error_test_dict, train_part)
    # 验证 G
    # elvation(state+' generator', 1, dh, step, sess, generator, merged, writer, valid_test_dict,
    #          error_test_dict, train_part)

