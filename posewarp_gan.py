import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import paDataReader
import datageneration
import networks
import scipy.io as sio
import param
import util
from keras.models import load_model,Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19


def createFeeds(params):

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')

	lift_warp_train,lift_warp_test = datareader.makeWarpExampleList(lift_params,20000,2000,2,1)
	golf_warp_train,golf_warp_test = datareader.makeWarpExampleList(golf_params,50000,5000,2,2)

	warp_train = lift_warp_train + golf_warp_train
	warp_test = lift_warp_test + golf_warp_test

	warp_train_feed = datageneration.warpExampleGenerator(warp_train,params)
	warp_test_feed = datageneration.warpExampleGenerator(warp_test,params)

	transfer_train_feed = datageneration.transferExampleGenerator(lift_warp_train,golf_warp_train,params,0.5)
	transfer_test_feed = datageneration.transferExampleGenerator(lift_warp_test,golf_warp_test,params,0.5)

	return warp_train_feed,warp_test_feed,transfer_train_feed,transfer_test_feed


def train(model_name,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name 
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	warp_train_feed,warp_test_feed,transfer_train_feed,transfer_test_feed = createFeeds(params)

	batch_size = params['batch_size']

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		disc_loss_weight = 0.1
		l2_loss_weight = 1.0
		disc_lr = 1e-4
		gan_lr = 1e-4

		with tf.device(gpu):
			vgg_model = VGG19(weights='imagenet',include_top=False,input_shape=(256,256,3))
			networks.make_trainable(vgg_model,False)
			generator = networks.network_warp(params,vgg_model)
			generator.load_weights('../results/networks/golf+lifting/35000.h5')

			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))

			gan_warp = networks.gan(generator,discriminator,params)
			gan_transfer = Model(gan_warp.inputs, gan_warp.outputs[-1])

			gan_warp.compile(optimizer=Adam(lr=gan_lr),loss=['mse','binary_crossentropy'],loss_weights=[l2_loss_weight,disc_loss_weight])
			gan_transfer.compile(optimizer=Adam(lr=gan_lr),loss='binary_crossentropy',loss_weights=[disc_loss_weight])

		'''
		for j in xrange(2):
			X_warp,Y_warp = next(warp_train_feed)
			#X_pose,Y_pose = next(pose_train_feed)
			#X_tran = next(transfer_train_feed)

		sio.savemat('gan.mat', {'X': X_warp[0], 'Y': Y_warp, 'X_pose': X_warp[1], 'X_mask': X_warp[4]})
		'''

		step = 0	
		while(True):


			if(step % 2 == 0):
				X,Y = next(warp_train_feed)
			else:
				X,Y = next(transfer_train_feed)

			with tf.device(gpu):
				gen = generator.predict(X[0:4])[0]	


			#Train discriminator
			networks.make_trainable(discriminator,True)	
			X_img_disc = np.concatenate((Y,gen))
			X_pose_disc = np.concatenate((X[1][:,:,:,14:],X[1][:,:,:,14:]))
			X_mask_disc = np.concatenate((X[4],X[4]))

			L = np.zeros([2*batch_size,2])
			L[0:batch_size,0] = 1
			L[batch_size:,1] = 1

			inputs = [X_img_disc,X_pose_disc] #,X_mask_disc]	
			d_loss = discriminator.train_on_batch(inputs,L)
			networks.make_trainable(discriminator,False)


			#TRAIN GAN
			L = np.zeros([batch_size,2])
			L[:,0] = 1 #Pretend these are real.

			if(step % 2 == 0):
				X,Y = next(warp_train_feed)
				g_loss = gan_warp.train_on_batch(X[0:4],[Y,L])
				util.printProgress(step,0,[g_loss[1],d_loss])
			else:
				X,Y = next(transfer_train_feed)
				gan_transfer.train_on_batch(X[0:4],L)


			'''
			#Test
			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(2)			
				for j in xrange(n_batches):	
					X_warp,Y_warp = next(warp_test_feed)
					L = np.zeros([batch_size,2])
					L[:,1] = 1 #Fake images

					test_loss_j = gan_warp.test_on_batch(X_warp, [Y_warp,L])
					test_loss += np.array(test_loss_j[1:3])
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)
			'''

			if(step % params['test_save_interval']==0):
				#X_warp,Y_warp = next(warp_test_feed)
				#pred_val = gan_warp.predict(X_warp)[0]
				#sio.savemat(output_dir + '/' + str(step) + '.mat',
         		#{'X': X_warp[0],'Y': Y_warp, 'pred': pred_val}) 

				X_trans,Y_trans = next(transfer_test_feed)
				pred_val = gan_warp.predict(X_trans[0:4])[0]
	
				sio.savemat(output_dir + '/' + str(step) + '_trans.mat',
         		{'X': X_trans[0],'pred': pred_val, 'Y': Y_trans}) 
			
			if(step % params['model_save_interval']==0): 
				gan_warp.save(network_dir + '/' + str(step) + '.h5')			
	
			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1], sys.argv[2])
