import tensorflow as tf
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
c = tf.stack([a,b],axis=1)
c2 = tf.stack([a,b],axis=0)

d = tf.unstack(c,axis=0)

e = tf.unstack(c,axis=1)
print(c.get_shape())
print(c2.get_shape())
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c2))
    print(sess.run(d))
    print(sess.run(e))