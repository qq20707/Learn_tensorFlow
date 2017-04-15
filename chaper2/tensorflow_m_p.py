#TensorFlow model persistence
import tensorflow as tf 

v1 = tf.Variable(tf.constant(4.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')

v1 = tf.placeholder(tf.float32,shape=[1],name='v1')
v2 = tf.placeholder(tf.float32,shape=[1],name='v2')
result = v1+v2

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(result,feed_dict={v1:[4.0],v2:[2.0]}))
	saver.save(sess,"/home/lhuan/Desktop/Project/Tensorflow/chaper2/path/to/model/model.ckpt")


