import tensorflow as tf

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x不是一个特定的值，而是一个占位符placeholder
# 这里的None表示此张量的第一个维度可以是任何长度的。
# 784表示一个图片的像素总数
# 表示一个输入源，二维组，第一维度不知道，第二维度是784.
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# 计算交叉熵
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化我们创建的变量
init = tf.initialize_all_variables()
# 在一个Session里面启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)

# 模型循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 随机抓取训练数据中的100个批处理数据点
    # 用这些数据点作为参数替换之前的占位符来运行train_step.？？
    # feed_dict  向…提供; 供…作食物
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 评估我们的模型
# tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal
# 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配）
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# [True, False, True, True] 会变成 [1,0,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))








