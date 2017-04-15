import tensorflow as tf
import os
import numpy as np
import sys
from keras import backend as K
import cv2
import datareader
import preprocess
import network
import scipy.io as sio
from keras.callbacks import History
from tensorflow.python.framework import ops

n_joints = 14
batch_size = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
training_iters = 500000
gpu = '/gpu:3'
test_interval = 1000
n_test_vids = 13

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.readExamples('../datasets/golfswinghd/videos/',
			'../datasets/golfswinghd/videoinfo/',13,25,5,'.jpg')

	queue_train = tf.train.slice_input_producer(ex_test,shuffle=True)

	param = {}
	stride = 4
	param['crop_size_x'] = 256
	param['crop_size_y'] = 256
	param['target_dist'] = 1.171
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['stride'] = stride
	param['sigma'] = 7

	'''
	x_tf = tf.placeholder(tf.float32, shape = (batch_size,IMG_HEIGHT,IMG_WIDTH))
	y_tf = tf.placeholder(tf.float32, shape = (batch_size,IMG_HEIGHT,IMG_WIDTH))
	im = tf.placeholder(tf.float32, shape = (batch_size,IMG_HEIGHT,IMG_WIDTH,3))

	out_tf = network.interpolate([im,x_tf,y_tf])
	'''

	hist = History()
	with tf.Session(config=config) as sess:
		K.set_session(sess)
	
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		#with tf.device(gpu):
		#	model,model2 = network.network_matching(n_joints,IMG_HEIGHT,IMG_WIDTH,3)

		X_img = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))
		#X_pose = np.zeros((batch_size, IMG_HEIGHT/stride, IMG_WIDTH/stride,n_joints*2))
		Y = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))

		step = 0
		while step < 1:
			print step
			for i in xrange(batch_size):
				ex = sess.run(queue_train)
				I0,I1,H0,H1,Hlimb,Ilimb = preprocess.makeInputOutputPair_spattransf(ex,param)
				#I0 = preprocess.makeInputOutputPair_spattransf(ex,param)
				#sio.savemat('test.mat', {'I1':I1, 'H': Hlimb, 'I0': I0, 'Ilimb': Ilimb})
				#X_img[i,:,:,:] = I0
				#X_pose[i,:,:,:] = np.concatenate((H0,H0), axis=2)	
				#Y[i,:,:,:] = I0
				sio.savemat('test.mat', {'I0': I0, 'I1': I1, 'Hlimb': Hlimb, 'Ilimb': Ilimb})

			step += 1
			
		coord.request_stop()
		coord.join(threads)
		sess.close()

if __name__ == "__main__":
	train()
