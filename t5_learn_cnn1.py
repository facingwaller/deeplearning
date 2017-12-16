# coding=utf-8
import tensorflow as tf
import numpy as np

input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # print(input)
    # print(input.eval())
    print("case 2")
    print(sess.run(op))


# scores = [ [0,10],[1,11]  ]
# predictions = tf.argmax(scores, 1, name="predictions")


