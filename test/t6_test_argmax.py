# http://blog.csdn.net/m0_37041325/article/details/77159660
# http://blog.csdn.net/qq575379110/article/details/70538051
import tensorflow as tf
x=tf.constant([[111.,1.,116],[6.,2.,6]])

xShape=tf.shape(x)
# tf.argmax(input, axis=None, name=None, dimension=None) 此函数是对矩阵按行或列计算最大值
# axis：0表示按列，1表示按行
z1=tf.arg_max(x,0)#沿axis=0操作
# [1 0 0]
# [2 0]

with tf.Session() as sess:
    xShapeValue,d1=sess.run([xShape,z1])
    print('shape= %s'%(xShapeValue))
    print(d1)