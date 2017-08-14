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
import myVGG
from keras.models import load_model,Model
from keras.optimizers import Adam


def createFeeds(params):

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

	warp_train_feed = datageneration.warpExampleGenerator(warp_train,params)
	warp_test_feed = datageneration.warpExampleGenerator(warp_test,params)

	#transfer_train_feed = datageneration.transferExampleGenerator(lift_warp_train,golf_warp_train,params,0.5)
	#transfer_test_feed = datageneration.transferExampleGenerator(lift_warp_test,golf_warp_test,params,0.5)

	return warp_train_feed,warp_test_feed #,transfer_train_feed,transfer_test_feed


def train(model_name,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	warp_train_feed,warp_test_feed = createFeeds(params)

	batch_size = params['batch_size']

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		gan_lr = 1e-4
		disc_lr = 1e-4
		disc_loss = 0.1

		with tf.device(gpu):
			vgg_model = myVGG.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')
			generator = networks.network_fgbg(params,vgg_model,response_weights)
			generator.load_weights('../results/networks/fgbg_small/100000.h5')

			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))
			gan_warp = networks.gan(generator,discriminator,params,vgg_model,response_weights,disc_loss,gan_lr)


		step = 0	
		while(True):

			X,Y = next(warp_train_feed)

			with tf.device(gpu):
				gen = generator.predict(X)	
				#mask = mask_model.predict(X)


			#Train discriminator
			networks.make_trainable(discriminator,True)	
			#mask = np.tile(X[-1], [2,1,1,1])
	
			X_img_disc = np.concatenate((Y,gen))# * np.tile(mask,[1,1,1,3])
			X_src_pose_disc = np.concatenate((X[1],X[1]))
			X_tgt_pose_disc = np.concatenate((X[2],X[2]))

			#sio.savemat('test.mat', {'X': X_img_disc, 'pose': X_tgt_pose_disc})
			#return

			L = np.zeros([2*batch_size,2])
			L[0:batch_size,0] = 1
			L[batch_size:,1] = 1

			inputs = [X_img_disc,X_src_pose_disc,X_tgt_pose_disc]
			d_loss = discriminator.train_on_batch(inputs,L)
			networks.make_trainable(discriminator,False)

			#Not sure this does anything, but I train the discriminator a couple
			#of iterations before starting the gan		
			if(step < 5):
				util.printProgress(step,0,[0,d_loss])
				step += 1
				continue
	
			#TRAIN GAN
			L = np.zeros([batch_size,2])
			L[:,0] = 1 #Pretend these are real.
			X,Y = next(warp_train_feed)
			g_loss = gan_warp.train_on_batch(X,[Y,L])
			util.printProgress(step,0,[g_loss[1],d_loss])

			#Test
			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(1)			
				for j in xrange(n_batches):	
					X_warp,Y_warp = next(warp_test_feed)
					test_loss += np.array(generator.test_on_batch(X_warp,Y_warp))
					#L = np.zeros([batch_size,2])
					#L[:,1] = 1 #Fake images

					#test_loss_j = gan_warp.test_on_batch(X_warp, [Y_warp,L])
					#test_loss += np.array(test_loss_j[1:3])
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['model_save_interval']==0): 
				gan_warp.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1], sys.argv[2])
