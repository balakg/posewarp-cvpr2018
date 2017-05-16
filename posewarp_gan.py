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

	pa_params = param.getDatasetParams('pennaction')
	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')

	#Warp train: penn, lift, golf
	pa_warp_train,pa_warp_test = paDataReader.makePAWarpExampleList(pa_params,60000,6000)
	lift_warp_train,lift_warp_test = datareader.makeWarpExampleList(lift_params,10000,1000,2)
	golf_warp_train,golf_warp_test = datareader.makeWarpExampleList(golf_params,20000,2000,2)

	warp_train = pa_warp_train + lift_warp_train + golf_warp_train
	warp_test = pa_warp_test + lift_warp_test + golf_warp_test

	#Pose train: all
	mpii_pose_train,mpii_pose_test = datareader.makePoseExampleList('json/MPII_annotations.json',0,14)
	leeds_pose_train,leeds_pose_test = datareader.makePoseExampleList('json/LEEDS_annotations.json',0,14)

	pose_train = mpii_pose_train + leeds_pose_train + [ex[0:32] for ex in warp_train]

	warp_train_feed = datageneration.warpExampleGenerator(warp_train,params)
	warp_test_feed = datageneration.warpExampleGenerator(warp_test,params)
	pose_train_feed = datageneration.poseExampleGenerator(pose_train,params)

	#transfer_train_feed = datageneration.transferExampleGenerator(lift_train,golf_train,params,0.5)
	#transfer_test_feed = datageneration.transferExampleGenerator(lift_test,golf_test,params,0.5)

	return warp_train_feed,warp_test_feed,pose_train_feed


def train(model_name,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name 
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	warp_train_feed,warp_test_feed,pose_train_feed = createFeeds(params)

	batch_size = params['batch_size']

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = VGG19(weights='imagenet',include_top=False,input_shape=(256,256,3))
			networks.make_trainable(vgg_model,False)
			generator = networks.network_warp(params,vgg_model)
			generator.load_weights('../results/networks/combined/90000.h5')

			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))

			gan_warp = networks.gan(generator,discriminator,params)
			#gan_transfer = Model(gan_warp.inputs, gan_warp.outputs[2])

			gan_warp.compile(optimizer=Adam(lr=1e-4),loss=['mse','mse','binary_crossentropy'],
						loss_weights=[1.0,0.001,0.01])

			#gan_transfer.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',loss_weights=[0.01])
	

		step = 0	
		while(True):
			X_warp,Y_warp = next(warp_train_feed)

			with tf.device(gpu):
				gen = generator.predict(X_warp)[0]	

			#Train discriminator
			networks.make_trainable(discriminator,True)
			
			X_pose,Y_pose = next(pose_train_feed)

			X_img_disc = np.concatenate((X_pose[0][0:batch_size/2,:,:,:],
										 Y_warp[0:batch_size/2,:,:,:],gen))

			X_pose_disc = np.concatenate((Y_pose[0:batch_size/2,:,:,:],
										 X_warp[1][0:batch_size/2,:,:,14:],
										 X_warp[1][:,:,:,14:]))

			L = np.zeros([2*batch_size,2])
			L[0:batch_size,0] = 1
			L[batch_size:,1] = 1

			inputs = [X_img_disc,X_pose_disc]	
			d_loss = discriminator.train_on_batch(inputs,L)
			networks.make_trainable(discriminator,False)

			#TRAIN GAN
			L = np.zeros([batch_size,2])
			L[:,0] = 1 #Pretend these are real.

			if(True): # step % 2 == 0):
				X_warp,Y_warp = next(warp_train_feed)
				Y_warp_vgg = vgg_model.predict(util.vgg_preprocess(Y_warp))
				g_loss = gan_warp.train_on_batch(X_warp,[Y_warp,Y_warp_vgg,L])
				util.printProgress(step,0,[g_loss[1],g_loss[2],d_loss])
			else:
				X_trans = next(transfer_train_feed)
				gan_transfer.train_on_batch(X_trans,L)

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(3)			
				for j in xrange(n_batches):	
					X_warp,Y_warp = next(warp_test_feed)
					L = np.zeros([batch_size,2])
					L[:,1] = 1 #Fake images

					Y_warp_vgg = vgg_model.predict(util.vgg_preprocess(Y_warp))
					test_loss_j = gan_warp.test_on_batch(X_warp, [Y_warp,Y_warp_vgg,L])
					test_loss += np.array(test_loss_j[1:4])
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			
			if(step % params['test_save_interval']==0):
				X_warp,Y_warp = next(warp_test_feed)
				pred_val = gan_warp.predict(X_warp)[0]
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X': X_warp[0],'Y': Y_warp, 'pred': pred_val}) 

				#X_trans = next(transfer_test_feed)
				#pred_val = gan_warp.predict(X_trans)[0]
	
				#sio.savemat(output_dir + '/' + str(step) + '_trans.mat',
         		#{'X': X_trans[0],'pred': pred_val}) 

			if(step % params['model_save_interval']==0): 
				gan_warp.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1], sys.argv[2])
