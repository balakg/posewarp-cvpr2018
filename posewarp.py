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
import paDataReader
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

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	yoga_params = param.getDatasetParams('yoga')

	lift_train,lift_test = datareader.makeWarpExampleList(lift_params,20000,2000,2,1)
	golf_train,golf_test = datareader.makeWarpExampleList(golf_params,50000,5000,2,2)
	yoga_train,yoga_test = datareader.makeWarpExampleList(yoga_params,12000,1200,2,3)

	warp_train = lift_train + golf_train + yoga_train
	warp_test = lift_test + golf_test + yoga_test

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
			vgg_model = VGG19(weights='imagenet',include_top=False,
						input_shape=(256,256,3))
			networks.make_trainable(vgg_model,False)
			model = networks.network_fgbg(params,vgg_model)

			#model = networks.network_ed(params,vgg_model)
			#model.load_weights('../results/networks/l2+vgg/8000.h5',by_name=True)
			model.compile(optimizer=Adam(lr=1e-4),loss=['mse','mse'],
						loss_weights=[1.0,0.001])		
			#model.compile(optimizer=Adam(lr=2e-4),loss=['mse'])


		#model.save(network_dir + '/' + str(0) + '.h5')			

		step = 0
		while(True):
			X,Y = next(train_feed)			

			with tf.device(gpu):
				Y_vgg = vgg_model.predict(util.vgg_preprocess(Y))
				train_loss = model.train_on_batch(X,[Y,Y_vgg])

			util.printProgress(step,0,train_loss)

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(3)
				for j in xrange(n_batches):	
					X,Y = next(test_feed)			
					Y_vgg = vgg_model.predict(util.vgg_preprocess(Y))
					test_loss += np.array(model.test_on_batch(X,[Y,Y_vgg]))

				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['test_save_interval']==0):
				X,Y = next(test_feed)			
				pred = model.predict(X)[0]
	
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
