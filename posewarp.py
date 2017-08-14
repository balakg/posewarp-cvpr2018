import tensorflow as tf
import os
import numpy as np
import sys
import datareader
import datageneration
import networks
import scipy.io as sio
import param
import util
from keras.models import load_model,Model
from keras.optimizers import Adam
import myVGG

def train(model_name,gpu_id):	

	params = param.getGeneralParams()

	gpu = '/gpu:' + str(gpu_id)

	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')

	lift_train,lift_test = datareader.makeWarpExampleList(lift_params,9000,1000,2,1)
	golf_train,golf_test = datareader.makeWarpExampleList(golf_params,22000,2500,2,2)
	workout_train,workout_test = datareader.makeWarpExampleList(workout_params,12500,1250,2,3)
	tennis_train,tennis_test = datareader.makeWarpExampleList(tennis_params,10000,1000,2,4)
	
	warp_train = lift_train + golf_train + workout_train + tennis_train
	warp_test = lift_test + golf_test + workout_test + tennis_test

	train_feed = datageneration.warpExampleGenerator(warp_train,params,return_pose_vectors=False)
	test_feed = datageneration.warpExampleGenerator(warp_test,params,return_pose_vectors=False)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = myVGG.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')
			model = networks.network_fgbg(params,vgg_model,response_weights,do_dropout=False,loss='vgg')
			#model.load_weights('../results/networks/fgbg_extraconv/197000.h5')

		#model.summary()
		#return

		'''
		step = 0
		mean_response = []
		std_response = []
		for step in xrange(1000):
			print step
			X,Y = next(train_feed)			
			pred_step = vgg_model.predict(util.vgg_preprocess(X[0]))

			for i in xrange(len(pred_step)):
				mean_step = np.mean(pred_step[i],axis=(0,1,2))
				std_step = np.std(pred_step[i],axis=(0,1,2))

				if(step == 0):
					mean_response.append(mean_step)
					std_response.append(std_step)
				else:
					mean_response[i] += mean_step
					std_response[i] += std_step

			step += 1		
		
		for i in xrange(len(mean_response)):
			mean_response[i]/= (1000.0)
			std_response[i]/= (1000.0)

		responses = {}
		for i in xrange(12):
			responses[str(i)] = (mean_response[i],std_response[i])

		sio.savemat('mean_response_new.mat', responses)

		return
		'''
		step = 0
		while(True):
			X,Y = next(train_feed)			
			'''
			if(step <= 200):
				X_switch = np.zeros((4,2,3,11))
			else:
				X_switch = np.ones((4,2,3,11))		
			X.append(X_switch)
			'''

			with tf.device(gpu):
				train_loss = model.train_on_batch(X,Y)

			util.printProgress(step,0,train_loss)
	
			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(1)
				for j in xrange(n_batches):	
					X,Y = next(test_feed)			
					test_loss += np.array(model.test_on_batch(X,Y))

				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['model_save_interval']==0):
				model.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	


if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1],sys.argv[2])
