import tensorflow as tf
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
c = tf.stack([a,b],axis=1)
c2 = tf.stack([a,b],axis=0) # 默认

d = tf.unstack(c2,axis=0)  # 默认

e = tf.unstack(c,axis=1)
print(c)
print(c2)
print(c.get_shape())
print(c2.get_shape())
print("=============================")
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c2))
    print("=============================")
    print(sess.run(d))
    # print(sess.run(e))