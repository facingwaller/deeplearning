import tensorflow as tf
import numpy as np

from lib.logistics_helper import logistics_helper
from lib.ct import ct

# X_SIZE = 784
# Y_SIZE = 10

X_SIZE = 2
Y_SIZE = 2

# 改成 2 位 ， 2分类


x1 = []
y1 = []

for i in range(8):
    t1 = float((i % 10 + 1) / 10)
    x1.append(t1)
    t2 = float(i % 2)
    y1.append(t2)

x1 = np.array(x1)
y1 = np.array(y1)

# mnist = mnist.input_data.read_data_sets('./data/mnist', one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='input')  # 784 个  像素
y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='output')  # 10 分类

w = tf.Variable(tf.random_normal([X_SIZE, Y_SIZE], 0, 0.05, dtype=tf.float32), name='weights')
b = tf.Variable(tf.random_normal([1, Y_SIZE], dtype=tf.float32), name='bias')

logits = tf.add(tf.matmul(x, w), b)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))

lr = 0.001
epochs = 100
batch_size = 10
f1='../data/nlpcc2016/8-logistics/logistics-2018-03-10.txt_bak.txt'
lh = logistics_helper(f1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

ct.print("lr %s size %s f1= %s"%(lr,batch_size,f1))

def run_step(data, model):
    # if epoch % 10 == 0:
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    gc_valid = lh.batch_iter(data, batch_size)
    for gc_valid_item in gc_valid:
        total += 1
        x1 = gc_valid_item[1]
        y1 = gc_valid_item[2]
        x1 = x1.reshape(-1, 2)
        y1 = y1.reshape(-1, 2)
        val_loss, val_acc,w1,b1 = sess.run([loss, accuracy,w,b],
                                     # feed_dict={x: mnist.validation.images, y: mnist.validation.labels}
                                     feed_dict={x: x1, y: y1}
                                     )
        if val_acc==0:
            ct.print(gc_valid_item[0],'error')
        total_loss += val_loss
        total_acc += val_acc

    ct.print('model %s epoch %s train loss = %s,train acc %s,the val loss=%s,val acc %s' %
             (model, epoch, train_loss, train_acc, total_loss / total, total_acc / total), model)
    ct.print("w1 %s"%(w1))
    ct.print("b1 %s"%(w1))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('graphs/logistic_reg', sess.graph)
    # n_batch = logistics_helper. # mnist.train.num_examples // batch_size # 取整除 - 返回商的整数部分

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        # gc1 = lh.batch_iter(lh.train_data,batch_size)
        # for _ in range(n_batch):
        # for gc1_item in gc1:
        #     x1 = gc1_item[1].reshape(-1,2)
        #     y1 = gc1_item[2].reshape(-1,2)
        #     _, batch_loss, batch_acc = sess.run([optimizer, loss, accuracy], feed_dict={x: x1, y: y1})
        #     train_loss += batch_loss
        #     train_acc += batch_acc
        run_step(lh.train_data, 'train')
        if epoch % (epochs // 10) == 0 and epoch != 0:
            run_step(lh.train_data, 'valid')
        if epoch % (epochs // 20) == 0 and epoch != 0:
            run_step(lh.test_data, 'test')

