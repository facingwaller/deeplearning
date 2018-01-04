import tensorflow as tf
var = tf.Variable(tf.random_uniform([2, 3],minval=0.001,maxval=1), name="var")
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
print(sess.run(var))

