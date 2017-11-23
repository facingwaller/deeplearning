import tensorflow as tf
import numpy as np
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x不是一个特定的值，而是一个占位符placeholder
#这里的None表示此张量的第一个维度可以是任何长度的。
#784表示一个图片的像素总数
#表示一个输入源，二维组，第一维度不知道，第二维度是784.
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

a = tf.constant(1.,name="const1")
b = tf.constant(2.,name="const2")
c = tf.add(a,b)
with tf.Session() as sess :
	print (sess.run(c))
	print (c.eval())