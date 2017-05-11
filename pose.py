import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import datageneration
import networks
import scipy.io as sio
import param
import util
import random
from keras.models import load_model,Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19

def train(model_name,gpu_id):	

	params = param.getGeneralParams()

	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)


	ex1,_ = datareader.makePoseExampleList('json/MPII_annotations.json',0,params['n_joints'])
	ex2,_ = datareader.makePoseExampleList('json/LEEDS_annotations.json',0,params['n_joints'])
	
	ex_all = ex1 + ex2
	random.shuffle(ex_all)
	ex_test = ex_all[0:500]
	ex_train = ex_all[1000:]

	train_feed = datageneration.poseExampleGenerator(ex_train,params)
	test_feed = datageneration.poseExampleGenerator(ex_test,params)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)


		with tf.device(gpu):
			model = networks.posenet(params)
			model.compile(optimizer=Adam(lr=1e-4),loss='mse')

		'''
		X,Y = next(test_feed)
		sio.savemat('test.mat', {'X': X[0], 'Y': Y, 'mask': X[1]})		
		'''
	
		step = 0	
		while(True):
			X,Y = next(train_feed)			

			with tf.device(gpu):
				train_loss = model.train_on_batch(X,Y)

			util.printProgress(step,0,train_loss)

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = 0
				for j in xrange(n_batches):	
					X,Y = next(test_feed)			
					test_loss += model.test_on_batch(X,Y)

				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['test_save_interval']==0):
				X,Y = next(test_feed)			
				pred = model.predict(X)
	
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X': X[0],'Y': Y, 'pred': pred})	

			if(step % params['model_save_interval']==0):
				model.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1],sys.argv[2])
