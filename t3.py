# coding=utf-8
# 简单神经网络实现自定义损失函数（利润最大化）
# 不包含隐层

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

y = tf.matmul(x, w1)

# 定义预测多了和少了的成本
loss_less = 10
loss_more = 1
# 自定义损失函数

# tf.select(p1, p2, p3)
# p1：是一个布尔类型的变量，比如True，也可以是一个表达式，返回值是True或者False。
# 这个变量可以是一个也可以是一个列表，就是很多个True或者False组成的列表
# 如果是True，返回p2，反之返回p3
# ensorflow1.0+版本中,select函数已经被where函数取代
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))

# 自定义损失函数+L2正则化损失，0.5是lamada参数
# 在损失函数上加上正则项是防止过拟合的一个重要方法
# https://www.cnblogs.com/linyuanzhou/p/6923607.html
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),
                    (y-y_)*loss_more,(y_-y)*loss_less))+tf.contrib.layers.l2_regularizer(0.5)(w1)

# 实现了Adam算法的优化器,Adam一种基于一阶梯度来优化随机目标函数的算法。
# tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9,
#  beta2=0.999, epsilon=1e-08, use_locking=False, name=’Adam’)
# https://www.cnblogs.com/xinchrome/p/4964930.html
# 以后可以考虑用更多别的优化算法来试试效果
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)


global_step = tf.Variable(0)
# 学习率的设置：指数衰减法，参数：初始参数，全局步骤，每训练100轮乘以衰减速度0,96(当staircase=True的时候)
# 实现指数衰减学习率 exponential 指数，decay 衰减，staircase 梯度
# decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
# learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96
# ···········
# 其中，decayed_learning_rate 为每一轮优化时使用的学习率；
#           learning_rate为事先设定的初始学习率；0.1
#           global_step，0
#           decay_steps为衰减速度。100
#           decay_rate为衰减系数；0.96
#
# tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
# Gradient梯度，Descent下降，Optimizer优化器
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 加入了一个噪音值，-0.05～0.05之间
# 重点 1*x1 + 1*x2 乘以的1 是实际的 参数

Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))

    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_loss = sess.run(
                loss, feed_dict={x: X, y_: Y})
            print("After %d training_step(s) ,loss on all data is %g" % (i, total_loss))
    print(sess.run(w1))

