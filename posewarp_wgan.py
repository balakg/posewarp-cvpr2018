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
from keras.optimizers import Adam,RMSprop
from keras.backend.tensorflow_backend import set_session

def createFeeds(params):

	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')
	
	golf_train,golf_test = datareader.makeWarpExampleList(golf_params,22000,2200,False,2)
	workout_train,workout_test = datareader.makeWarpExampleList(workout_params,12500,1250,False,3)
	tennis_train,tennis_test = datareader.makeWarpExampleList(tennis_params,10000,1000,False,4)

	warp_train = golf_train + workout_train + tennis_train
	warp_test = golf_test + workout_test + tennis_test

	warp_train_feed = datageneration.warpExampleGenerator(warp_train,params)
	warp_test_feed = datageneration.warpExampleGenerator(warp_test,params)

	return warp_train_feed,warp_test_feed


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
	set_session(tf.Session(config=config))	

	gan_lr = 5e-5
	disc_lr = 5e-5
	disc_loss = 0.1

	vgg_model_num = 120000

	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')
		generator = networks.network_fgbg(params,vgg_model,response_weights)
		generator.load_weights('../results/networks/fgbg_vgg/' + str(vgg_model_num) + '.h5')

		discriminator = networks.discriminator(params)
		discriminator.compile(loss=networks.wass,optimizer=RMSprop(disc_lr))
		gan = networks.gan(generator,discriminator,params,vgg_model,response_weights,disc_loss,gan_lr)


	for step in xrange(vgg_model_num+1,vgg_model_num+5001):
		for j in xrange(2):
			for l in discriminator.layers:
				weights = l.get_weights()
				weights = [np.clip(w, -0.01, 0.01) for w in weights]
				l.set_weights(weights)

			X,Y = next(warp_train_feed)

			with tf.device(gpu):
				gen = generator.predict(X)	

			#Train discriminator
			networks.make_trainable(discriminator,True)	

			X_tgt_img_disc = np.concatenate((Y,gen))
			X_src_pose_disc = np.concatenate((X[1],X[1]))
			X_tgt_pose_disc = np.concatenate((X[2],X[2]))

			L = np.ones(2*batch_size)
			L[0:batch_size] = -1

			inputs = [X_tgt_img_disc,X_src_pose_disc,X_tgt_pose_disc]
			d_loss = discriminator.train_on_batch(inputs,L)
			networks.make_trainable(discriminator,False)
	
		#TRAIN GAN
		L = -1*np.ones(batch_size)
		X,Y = next(warp_train_feed)
		g_loss = gan.train_on_batch(X,[Y,L])
		util.printProgress(step,0,[g_loss[1],d_loss])

		if(step % params['model_save_interval']==0): 
			gan.save(network_dir + '/' + str(step) + '.h5')			

		'''
		#Test
		if(step % params['test_interval'] == 0):
			n_batches = 16
			test_loss = np.zeros(1)
			for j in xrange(n_batches):	
				X,Y = next(warp_test_feed)			
				test_loss += np.array(generator.test_on_batch(X,Y))
	
			test_loss /= (n_batches)
			util.printProgress(step,1,[test_loss,0.0])
		'''

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1], sys.argv[2])
