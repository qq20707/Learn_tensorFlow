import tensorflow as tf 
import openpyxl
from numpy.random import RandomState
import numpy as np 

dataset_size =715

batch_size=100

INPUT_NODE = 2 # Input node
OUTPUT_NODE = 4 # Output node

LAYER1_NODE = 10

BATCH_SIZE = 100


LEARNING_RATE_BATE = 0.8  # Learn rate
LEARNING_RATE_DECAY = 0.99

REGULARZATION_RATE = 0.0001 
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY =0.99

XX=np.array([80.7 ,122.8])
YY=np.array([[0 ,0 ,0 ,1]])

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

		return tf.matmul(layer1,weights2) + biases2
	else:

		layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))

		return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)  

def train(inputDatsets):

	X=inputDatsets['inputX']
	Y=inputDatsets['inputY']
	#print(Y)
	XX = tf.placeholder(tf.float64,shape=(None,INPUT_NODE),name='x-input')

	x = tf.placeholder(tf.float32,shape=(None,INPUT_NODE),name='x-input')
	y_ = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE),name='y-input')
	#print(x)
	weights1 = tf.Variable(
		tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1),tf.float64)

	biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]),tf.float64)

	weights2 = tf.Variable(
		tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1),tf.float64)
	biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]),tf.float64)

	y = inference(x,None,weights1,biases1,weights2,biases2)

	global_step = tf.Variable(0,trainable =False)

	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY,global_step)

	variable_averages_op = variable_averages.apply(
		tf.trainable_variables())

	average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

	#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	#	y,tf.argmax(y_,1))
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		y,tf.argmax(y_,1))

	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)

	regularization = regularizer(weights1) + regularizer(weights2)

	loss = cross_entropy_mean + regularization

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BATE,
		global_step,
		BATCH_SIZE,
		LEARNING_RATE_DECAY)

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	train_op = tf.group(train_step,variable_averages_op)


	correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		validate_feed = {x:X,y_:Y}

		test_feed = {x:X[0:200],y_:Y[0:200]}

		t_feed={x:X[10:700]}
		print("first value:")
		print(sess.run(weights1))
		print(sess.run(weights2))

		for i in range(TRAINING_STEPS):
			if i % 1000 == 0:
				validate_acc = sess.run(accuracy,feed_dict=validate_feed)
				print("After %d training step(s),validation accuracy using average model is %g" % (i,validate_acc))

			start=(i*batch_size)%dataset_size
		
			end = min(start+batch_size,dataset_size)
		
		#sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
			xs = X[start:end]
			ys = Y[start:end] 

			sess.run(train_op,feed_dict={x: xs,y_: ys})

		test_acc = sess.run(accuracy,feed_dict=test_feed)

		print("After %d training step(s) ,test accuracy using average model is %g" % (TRAINING_STEPS,test_acc))
		print("Last value:")
		
		weights1=(sess.run(weights1))
		print(weights1)
		weights2=(sess.run(weights2))
		print(weights2)
		print("biase:")
		biases1=(sess.run(biases1))
		print(biases1)
		biases2=(sess.run(biases2))
		print(biases2)
	#	test_acc = sess.run(accuracy,feed_dict=t_feed)
	#	print("After %d training step(s) ,test accuracy using average model is %g" % (TRAINING_STEPS,test_acc))
		 
		#y = inference(XX,None,weights1,biases1,weights2,biases2)
		print("yce:")
		print(sess.run(average_y,feed_dict=t_feed))

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
	#print(Y)
	return DataSets
def evalute(TestDatasets):

	x = tf.placeholder(tf.float64,[None,INPUT_NODE],name="x-input")
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

	validate_feed = {x:XX}

	y = inference(x,None)

	with tf.Session() as sess:
		sess.run(y,feed_dict=validate_feed)
		print(y)


def main(argv=None):

	inputDatsets = readData()
	#print(inputDatsets)

	train(inputDatsets)
	#y = inference(XX,None,weights1,biases1,weights2,biases2)
	#print(sess.run(y))
	#evalute(inputDatsets)

if __name__ == '__main__':

	tf.app.run()