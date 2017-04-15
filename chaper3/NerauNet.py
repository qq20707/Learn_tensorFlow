import tensorflow as tf 
import openpyxl
from numpy.random import RandomState
import numpy as np 
def readData():

	wb = openpyxl.load_workbook('example.xlsx')
	sheet = wb.get_sheet_by_name('Sheet1')
	max_rows=sheet.max_row
	for row in range(2,max_rows+1):
		code=sheet['D'+str(row)].value
		sheet['E'+str(row)]=list(str(code))[0]
		sheet['F'+str(row)]=list(str(code))[1]
		sheet['G'+str(row)]=list(str(code))[2]
		sheet['H'+str(row)]=list(str(code))[3]

	Lux=[]
	Lx=[]
	ELx=[]
	for row in range(2,max_rows+1):
		ELx=sheet['B'+str(row)].value 
		Lx = sheet['C'+str(row)].value
		Lux = Lux +[[ELx,Lx]]

	Dlux=np.array(Lux)
	X=Dlux
	yy1=[]
	yy2=[]
	yy3=[]
	yy4=[]
	Y=[]
	for row in range(2,max_rows+1):
		yy1=int(sheet['E'+str(row)].value)
		yy2=int(sheet['F'+str(row)].value)
		yy3=int(sheet['G'+str(row)].value)
		yy4=int(sheet['H'+str(row)].value)
		Y=Y+[[yy1,yy2,yy3,yy4]]
	Y=np.array(Y)
	DataSets={'inputX':X,'inputY':Y}
	return DataSets



Datasets=readData()
batch_size =100
#print(Datasets)
X=Datasets['inputX']
Y=Datasets['inputY']

w1 = tf.Variable(tf.random_normal([2,10],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([10,4],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,4),name='y-input')

a = tf.nn.relu(tf.matmul(x,w1)+1)
y = tf.nn.relu(tf.matmul(a,w2)+1)

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

#rdm = RandomState(1)
dataset_size =715
#X = rdm.rand(dataset_size,2)
#Y = [[int(x1+x2<1)] for (x1,x2) in X]

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
	#sess=tf.Session()
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))

	STEPS = 5000
	for i in range(STEPS):
	
		start=(i*batch_size)%dataset_size
		
		end = min(start+batch_size,dataset_size)
		
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 1000 == 0:
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
			print("After %d training steps,cross entropy on all data is %g"% (i,total_cross_entropy))

	print(sess.run(w1))
	print(sess.run(w2))