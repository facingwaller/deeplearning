import tensorflow as tf
import numpy as np

from lib.logistics_helper import logistics_helper
from lib.ct import ct

# X_SIZE = 784
# Y_SIZE = 10

X_SIZE = 2
Y_SIZE = 2

# 改成 2 位 ， 2分类
lr = 0.001  # 0.001
epochs = 100
batch_size = 10
# f1='../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'
f1 = '../data/nlpcc2016/8-logistics/logistics-2018-03-20.txt_bak.txt'
#
lh = logistics_helper(f1)

# ----------------------

x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='input')  # 784 个  像素
y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='output')  # 10 分类
right_index = tf.placeholder(dtype=tf.int32, name='right_index')  # 正确的分类的index
w = tf.Variable(tf.random_normal([X_SIZE, Y_SIZE], 0, 0.05, dtype=tf.float32), name='weights')
b = tf.Variable(tf.random_normal([1, Y_SIZE], dtype=tf.float32), name='bias')

z = tf.add(tf.matmul(x, w), b)  # 得分
z1 = tf.reduce_mean(z, 1)  # 取均值
z_max = tf.reduce_max(z1)  # 取出最大的哪项
# # 正确分数
z_right = tf.gather(z1, right_index)
loss = tf.maximum(0.0,tf.reduce_max(z_right) - z_max)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
accuracy = tf.equal(loss, 0)

# logits = tf.reduce_mean(tf.add(tf.matmul(x, w), b))
# logits = z1  # tf.add(tf.matmul(x, w), b)
#
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))
# logits_step1 = tf.argmax(logits, 1)  # 最接近1的# tf.argmax就是返回最大的那个数值所在的下标
# logits_step1_5 = tf.argmax(y, 1)
# logits_step2 = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# logits_step3 = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32)
#
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
# optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# test_x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='test_input')  # 784 个  像素
# test_y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='test_output')  # 10 分类
# test_logits = tf.add(tf.matmul(test_x, w), b)
# test_accuracy = \
#     tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_y, 1)), tf.float32))
# test_logits_step1 = tf.argmax(test_logits, 1)  # 最接近1的
# test_logits_step1_5=tf.argmax(test_y, 1)
# test_logits_step2 = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_y, 1))
# test_logits_step3 = tf.cast(tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_y, 1)), tf.float32)

ct.print("lr %s size %s f1= %s" % (lr, batch_size, f1))


def run_step1(data, model):
    # if epoch % 10 == 0:
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    error_count = 0
    right_count = 0
    # ct.print("属性\t输入X\t输入答案\t判断答案\t\t\t\t", 'train')

    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        p1 = gc_valid_item[3]
        r_i = gc_valid_item[4]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)

        _z, _z1, _z_max, _z_right, _loss, _w, _b, _accuracy = \
            sess.run([z, z1, z_max, z_right, loss, w, b, accuracy],
                     feed_dict={x: x1, y: y1, right_index: r_i})
        if _accuracy == 0:

            right_count += 1
        else:
            # ct.print("%s\t\t\t\t\t\t" % gc_valid_item[0][0], 'train')
            error_count += 1
        # for index in range(len(l2)):
        #     item = l2[index]
        #     ct.print("%s\t%s\t%s\t%s" % (
        #         p1[index], x1[index], y1[index], l2[index]),
        #              'train')
        #     if not item:  # 出错的话 打印出来 and gc_valid_item[2][index][0] == 1
        #         ct.print("%s\t%s" % (p1[index], gc_valid_item[1][index]), 'error')  #
        #         error_count += 1

        # ct.print(gc_valid_item[0], 'error')
        # if val_acc == 0:
        #     ct.print(gc_valid_item[0], 'error')
        total_loss += _loss
        total_acc += _accuracy
        # ct.print("%s    %s"%(_loss,_accuracy))


    ct.print(
        'model %s epoch %s train loss = %s, val acc %s  error_count %s right_count %s total:%s' %
        (model, epoch, total_loss, total_acc / total,error_count,right_count, total),
        model + '_acc')
    return w, b


def run_step2(data, model):
    # if epoch % 10 == 0:
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    error_count = 0
    ct.print("属性\t输入X\t输入答案\t判断答案\t\t\t\t", 'train')

    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        p1 = gc_valid_item[3]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)

        ## 测试

        _z, _z1 = \
            sess.run([z, z1],

                     feed_dict={x: x1, y: y1})
        print('test')
        ##
        val_loss, val_acc, w1, b1, l0, l1, l1_5, l2, l3 = \
            sess.run([loss, accuracy, w, b, logits, logits_step1, logits_step1_5, logits_step2, logits_step3],
                     # feed_dict={x: mnist.validation.images, y: mnist.validation.labels}
                     feed_dict={x: x1, y: y1})
        if gc_valid_item[0][0] == '♠是用什么语言写的':
            print(1)
        ct.print("%s\t\t\t\t\t\t" % gc_valid_item[0][0], 'train')
        for index in range(len(l2)):
            item = l2[index]
            ct.print("%s\t%s\t%s\t%s" % (
                p1[index], x1[index], y1[index], l2[index]),
                     'train')
            if not item:  # 出错的话 打印出来 and gc_valid_item[2][index][0] == 1
                ct.print("%s\t%s" % (p1[index], gc_valid_item[1][index]), 'error')  #
                error_count += 1

                # ct.print(gc_valid_item[0], 'error')
        # if val_acc == 0:
        #     ct.print(gc_valid_item[0], 'error')
        total_loss += val_loss
        total_acc += val_acc

    acc_real = 1 - error_count / total
    ct.print(
        'model %s epoch %s train loss = %s,train acc %s,the val loss=%s,val acc %s acc_real=%s error_count:%s total:%s' %
        (model, epoch, train_loss, train_acc, total_loss / total, total_acc / total, acc_real, error_count, total),
        model + '_acc')
    # ct.print("w1=%s b1=%s"%(w1,b1))
    return w1, b1


def run_step_test(data, model):
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    error_count = 0

    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        p1 = gc_valid_item[3]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)
        val_acc, w1, b1, l0, l1, l2, l3 = \
            sess.run([test_accuracy, w, b, test_logits,
                      test_logits_step1, test_logits_step2, test_logits_step3],
                     # feed_dict={x: mnist.validation.images, y: mnist.validation.labels}
                     feed_dict={test_x: x1, test_y: y1})
        if gc_valid_item[0][0] == '♠这个游戏是什么公司发行的吗':
            print(1)

        ct.print(gc_valid_item[0][0], 'test')
        for index in range(len(l2)):
            item = l2[index]
            # X1输入  Y1输入    判断的答案   标准答案    判断的logistics
            ct.print("%s\t%s\t%s\t%s\t%s" % (x1[index], y1[index], gc_valid_item[2][index][0], item, l1), 'test')
            if not item and gc_valid_item[2][index][0] == 1:  # 出错的话 打印出来
                ct.print(gc_valid_item[1][index], 'error')  #
                error_count += 1

                # ct.print(gc_valid_item[0], 'error')
        # if val_acc == 0:
        #     ct.print(gc_valid_item[0], 'error')

        total_acc += val_acc

    acc_real = 1 - error_count / total
    ct.print(
        'model %s epoch %s train loss = %s,train acc %s,val acc %s acc_real=%s error_count:%s total:%s' %
        (model, epoch, train_loss, train_acc, total_acc / total, acc_real, error_count, total),
        model + '_acc')
    # ct.print("w1=%s b1=%s"%(w1,b1))
    return w1, b1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('graphs/logistic_reg', sess.graph)
    # n_batch = logistics_helper. # mnist.train.num_examples // batch_size # 取整除 - 返回商的整数部分

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        run_step1(lh.train_data, 'train')
        # if epoch % (epochs // 10) == 0 and epoch != 0:
        # run_step(lh.train_data, 'valid')
        # w1, b1 = run_step_test(lh.test_data, 'test')
        # if epoch % (epochs // 10) == 0 and epoch != 0:
        #     ct.print("w1=%s b1=%s" % (w1, b1))
