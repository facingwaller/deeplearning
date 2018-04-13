# 改成线性回归先用着
#
import tensorflow as tf
import numpy as np

from lib.logistics_helper import logistics_helper
from lib.ct import ct

# X_SIZE = 784
# Y_SIZE = 10

X_SIZE = 2
Y_SIZE = 2

# 改成 2 位 ， 2分类
lr = 0.01  # 0.001
epochs = 100
batch_size = 10
f1='../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'
# f1 = '../data/nlpcc2016/8-logistics/logistics-2018-03-20.txt_bak.txt'
#
lh = logistics_helper(f1)

# ----------------------

x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='input')  # 784 个  像素
y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='output')  # 10 分类
right_index = tf.placeholder(dtype=tf.int32, name='right_index')  # 正确的分类的index
# w = tf.Variable(tf.random_normal([X_SIZE, Y_SIZE], seed=1, dtype=tf.float32), name='weights',trainable=True)
# tf.fill([2,3], 9)
w = tf.Variable(tf.random_uniform(shape=[X_SIZE, Y_SIZE], minval=0, maxval=2, dtype=tf.float32), name='weights',
                trainable=True)
# w = tf.Variable([[ 1.0],[1.0]], name='weights',trainable=True)
b = tf.Variable(tf.random_normal([1, Y_SIZE], dtype=tf.float32), name='bias', trainable=True)

z = tf.add(tf.matmul(x, w), b)  # 得分
z1 = tf.reduce_mean(z, 1)  # 取均值
z_max = tf.reduce_max(z1)  # 取出最大的项
# # 正确分数
z_right = tf.gather(z1, right_index)
# loss = tf.maximum(0.0, tf.reduce_max(z_right) - z_max)
# loss = tf.multiply(z_max - tf.reduce_max(z_right),10000)  # loss = 最大者 与 正确答案的差距
loss = z_max - tf.reduce_max(z_right)  # loss = 最大者 与 正确答案的差距
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
accuracy = tf.equal(loss, 0)

test_x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='test_input')  # 784 个  像素
test_y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='test_output')  # 10 分类
test_right_index = tf.placeholder(dtype=tf.int32, name='test_right_index')  # 正确的分类的index
test_z = tf.add(tf.matmul(test_x, w), b)  # 得分
test_z1 = tf.reduce_mean(test_z, 1)  # 取均值
test_z_max = tf.reduce_max(test_z1)  # 取出最大的项
# # 正确分数
test_z_right = tf.gather(test_z1, test_right_index)
# loss = tf.maximum(0.0, tf.reduce_max(z_right) - z_max)
# loss = tf.multiply(z_max - tf.reduce_max(z_right),10000)  # loss = 最大者 与 正确答案的差距
test_loss = test_z_max - tf.reduce_max(test_z_right)  # loss = 最大者 与 正确答案的差距
test_accuracy = tf.equal(test_loss, 0)

ct.print("lr %s size %s f1= %s" % (lr, batch_size, f1))


def run_step1(data, model):
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    error_count = 0
    right_count = 0

    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        p1 = gc_valid_item[3]
        r_i = gc_valid_item[4]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)

        _z, _z1, _z_max, _z_right, _loss, _w, _b, _accuracy, _optimizer = \
            sess.run([z, z1, z_max, z_right, loss, w, b, accuracy, optimizer],
                     feed_dict={x: x1, y: y1, right_index: r_i})

        ct.print(_w, 'w')
        ct.print(_b, 'b')
        if _accuracy:
            right_count += 1
        else:
            error_count += 1
            ct.print(gc_valid_item[1], 'error')

        # print(_loss)
        total_loss += _loss
        total_acc += _accuracy
    ct.print(
        'model %s epoch %s   loss = %s,  acc = %s error_count %s right_count %s total:%s' %
        (model, epoch, total_loss / total, right_count / total, error_count, right_count, total),
        'debug')
    ct.print(_w, 'w')
    ct.print(_b, 'b')
    return w, b


def test_step1(data, model):
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    error_count = 0
    right_count = 0

    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        p1 = gc_valid_item[3]
        r_i = gc_valid_item[4]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)

        _z, _z1, _z_max, _z_right, _loss, _w, _b, _accuracy = \
            sess.run([test_z, test_z1, test_z_max, test_z_right, test_loss, w, b, test_accuracy],
                     feed_dict={test_x: x1, test_y: y1, test_right_index: r_i})

        ct.print(_w, 'w')
        ct.print(_b, 'b')
        if _accuracy:

            right_count += 1
        else:
            error_count += 1
        # print(_loss)
        total_loss += _loss
        total_acc += _accuracy
    ct.print(
        'model %s epoch %s   loss = %s,  acc = %s error_count %s right_count %s total:%s' %
        (model, epoch, total_loss / total, right_count / total, error_count, right_count, total),
        'debug')
    ct.print(_w, 'w')
    ct.print(_b, 'b')
    return w, b


ct.print("train baseline :%s" % lh.baseline(lh.train_data), 'debug')
ct.print("test baseline :%s" % lh.baseline(lh.test_data), 'debug')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('graphs/logistic_reg', sess.graph)
    # n_batch = logistics_helper. # mnist.train.num_examples // batch_size # 取整除 - 返回商的整数部分

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        w1, b1 = run_step1(lh.train_data, 'train')
        # if epoch % (epochs // 10) == 0 and epoch != 0:
        # run_step(lh.train_data, 'valid')
        test_step1(lh.test_data, 'test')
        if epoch % (epochs // 10) == 0 and epoch != 0:
            ct.print("w1=%s b1=%s" % (w1, b1))
