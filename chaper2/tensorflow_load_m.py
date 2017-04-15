#TensorFlow model persistence
import tensorflow as tf 
'''
v1 = tf.Variable(tf.constant(3.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(4.0,shape=[1]),name='v2')
#v1 = tf.placeholder(tf.float32,shape=[1],name='v12')
#v2 = tf.placeholder(tf.float32,shape=[1],name='v22')
result = v1*v2

#init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
	#sess.run(init_op)
	#saver.save(sess,"/home/lhuan/Desktop/Project/Tensorflow/chaper2/path/to/model/model.ckpt")
	saver.restore(sess,"/home/lhuan/Desktop/Project/Tensorflow/chaper2/path/to/model/model.ckpt")
	#print(sess.run(result,feed_dict={v1:[4],v2:[4]}))
	print(sess.run(result))
for i in range(1000):
	if i % 200 ==0:
		print(i)
'''
v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
with tf.Session() as sess:
	print(tf.clip_by_value(v,2.5,4.5).eval())