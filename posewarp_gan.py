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
from keras.models import load_model,Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19

def train(dataset,model_name,gpu_id):	

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name 
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	ex_train,ex_test = datareader.makeWarpExampleList(params)

	train_feed = datageneration.warpExampleGenerator(ex_train,params)
	test_feed = datageneration.warpExampleGenerator(ex_test,params)

	batch_size = params['batch_size']

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			#vgg_model = VGG19(weights='imagenet',include_top=False,
			#					input_shape=(128,128,3))
			#networks.make_trainable(vgg_model,False)
			generator = networks.network_warp(params)
			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-4))
			gan = networks.gan(generator,discriminator,params)
			gan.compile(optimizer=Adam(lr=5e-4),loss=['mse','binary_crossentropy'],
						loss_weights=[1.0,0.001])

		step = 0	
		while(True):
			X,Y = next(train_feed)

			with tf.device(gpu):
				gen = generator.predict(X)	
	
			#Train discriminator
			networks.make_trainable(discriminator,True)
			#Make a batch of batch_size generated examples and batch_size 
			#real examples. Each example is a source/target pair of images and poses.
			X_src_disc = np.concatenate((X[0],X[0]))
			X_tgt_disc = np.concatenate((gen,Y))
			X_pose_disc = np.concatenate((X[1],X[1]))

			L = np.zeros([2*batch_size,2])
			L[0:batch_size,1] = 1
			L[batch_size:,0] = 1

			inputs = [X_src_disc,X_tgt_disc,X_pose_disc]
			d_loss = discriminator.train_on_batch(inputs,L)

			networks.make_trainable(discriminator,False)


			#TRAIN GAN
			X,Y = next(train_feed)			
			L = np.zeros([batch_size,2])
			L[:,0] = 1 #Pretend these are real.

			g_loss = gan.train_on_batch(X,[Y,L])[1]

			util.printProgress(step,0,[g_loss,d_loss])

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(2)			
				for j in xrange(n_batches):	
					X,Y = next(test_feed)			
					L = np.zeros([batch_size,2])
					L[:,1] = 1 #Fake images

					test_loss_j = gan.test_on_batch(X, [Y,L])
					test_loss += np.array([test_loss_j[1],test_loss_j[2]])
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['test_save_interval']==0):
				X,Y = next(test_feed)			
				pred_val = gan.predict(X)
	
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X': X[0],'Y': Y, 'pred': pred_val[0]}) 

			
			if(step % params['model_save_interval']==0): 
				gan.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train('golfswinghd', sys.argv[1], sys.argv[2])
