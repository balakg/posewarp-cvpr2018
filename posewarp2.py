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

	lift_train,lift_test = datareader.makeWarpExampleList(lift_params,10000,1000,2,1)
	golf_train,golf_test = datareader.makeWarpExampleList(golf_params,25000,2500,2,2)
	workout_train,workout_test = datareader.makeWarpExampleList(workout_params,12500,1250,2,3)
	tennis_train,tennis_test = datareader.makeWarpExampleList(tennis_params,10000,1000,2,4)

	warp_train = lift_train + golf_train + workout_train + tennis_train
	warp_test = lift_test + golf_test + workout_test + tennis_test

	train_feed = datageneration.warpExampleGenerator(warp_train,params)
	test_feed = datageneration.warpExampleGenerator(warp_test,params)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			model = networks.motionNet(params)

		step = 0
		while(True):
			X,Y = next(train_feed)			

			Y = Y * np.tile(X[-1],[1,1,1,3])

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
