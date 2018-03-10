import tensorflow as tf
import numpy as np
import random

# X_SIZE = 784
# Y_SIZE = 10

X_SIZE = 2
Y_SIZE = 2

# 改成 2 位 ， 2分类

# 大于0.8 的就是
x1 = []
y1 = []

for i in range(8):
    t1 = float((i % 10 + 1) / 10)
    x1.append(t1)
    t2 = float(i % 2)
    y1.append(t2)

x1 = np.array(x1)
y1 = np.array(y1)

from tensorflow.examples.tutorials import mnist

mnist = mnist.input_data.read_data_sets('./data/mnist', one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, X_SIZE], name='input')  # 784 个  像素
y = tf.placeholder(dtype=tf.float32, shape=[None, Y_SIZE], name='output')  # 10 分类

w = tf.Variable(tf.random_normal([X_SIZE, Y_SIZE], 0, 0.05, dtype=tf.float32), name='weights')
b = tf.Variable(tf.random_normal([1, Y_SIZE], dtype=tf.float32), name='bias')

logits = tf.add(tf.matmul(x, w), b)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))

lr = 0.01
epochs = 100
batch_size = 2
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('graphs/logistic_reg', sess.graph)
    n_batch = mnist.train.num_examples // batch_size
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for _ in range(n_batch):
            z1, z2 = mnist.train.next_batch(batch_size)
            # print(type(z1))
            # print(type(z1[0]))
            # print(type(z2))
            # xs, ys = mnist.train.next_batch(batch_size)
            xs = x1
            ys = y1

            # batch_size = 2
            # wei_du = 2
            # max_document_length
            # word_d
            #         x_train = x_train.reshape((batch_size, max_document_length, word_d))
            # print(x1.shape)
            x1 = x1.reshape(-1,2)
            y1 = y1.reshape(-1,2)
            # print(x1.shape)
            # x_image = tf.reshape(x, [-1, 28, 28, 1])
            # print('......................')
            # print(x1)
            # print(y1)
            _, batch_loss, batch_acc = sess.run([optimizer, loss, accuracy], feed_dict={x: x1, y: y1})
            train_loss += batch_loss
            train_acc += batch_acc

        if epoch % 10 == 0:
            val_loss, val_acc = sess.run([loss, accuracy],
                                         # feed_dict={x: mnist.validation.images, y: mnist.validation.labels}
                                         feed_dict={x: x1, y: y1}
                                         )

            print('epoch {} train loss = {},train acc {},the val loss={},val acc {}' \
                  .format(epoch, train_loss / n_batch, train_acc / n_batch, val_loss, val_acc))

    test_loss, test_acc = sess.run([loss, accuracy],
                                   # feed_dict={x: mnist.test.images, y: mnist.test.labels}
                                   feed_dict={x: x1, y: y1}
                                   )

    print('the model on test loss is {},the accuracy is {}'.format(test_loss, test_acc))
    # writer.close()
