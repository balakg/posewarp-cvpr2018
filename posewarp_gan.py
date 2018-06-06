import tensorflow as tf
import os
import numpy as np
import sys
import datageneration
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.models import load_model,Model
from keras.optimizers import Adam

def train(model_name,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	train_feed=datageneration.create_feed(params, "train_vids.txt")
	test_feed=datageneration.create_feed(params, "test_vids.txt")

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
			vgg_model = truncated_vgg.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')
			#generator = networks.network_pix2pix(params,vgg_model,response_weights)
			generator = networks.network_posewarp(params)
			generator.load_weights('../results/networks/fgbg_vgg/100000.h5')

			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))
			gan = networks.gan(generator,discriminator,params,vgg_model,response_weights,disc_loss,gan_lr)
			gan.compile(optimizer=Adam(lr=gan_lr), loss=[networks.vgg_loss(vgg_model, response_weights), 'binary_crossentropy'],
                        loss_weights=[1.0,disc_loss])

		for step in xrange(10001):

			X,Y = next(train_feed)

			with tf.device(gpu):
				gen = generator.predict(X) #[0:3])	

			#Train discriminator
			X_tgt_img_disc = np.concatenate((Y,gen))
			X_src_pose_disc = np.concatenate((X[1],X[1]))
			X_tgt_pose_disc = np.concatenate((X[2],X[2]))

			L = np.zeros([2*batch_size])
			L[0:batch_size] = 1

			inputs = [X_tgt_img_disc,X_src_pose_disc,X_tgt_pose_disc]
			d_loss = discriminator.train_on_batch(inputs,L)

			#Train the discriminator a couple of iterations before starting the gan		
			if(step < 5):
				util.printProgress(step,0,[0,d_loss])
				step += 1
				continue
	
			#TRAIN GAN
			L = np.ones([batch_size])
			X,Y = next(train_feed)
			g_loss = gan.train_on_batch(X,[Y,L])
			util.printProgress(step,0,[g_loss[1],d_loss])


			'''
			#Test
			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(2)			
				for j in xrange(n_batches):	
					X,Y = next(warp_test_feed)
					#test_loss += np.array(generator.test_on_batch(X_warp,Y_warp))
					L = np.zeros([batch_size,2])
					L[:,1] = 1 #Fake images

					test_loss_j = gan_warp.test_on_batch(X_warp, [Y_warp,L])
					test_loss += np.array(test_loss_j[1:3])
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)
			'''

			if(step % params['model_save_interval']==0 and step > 0): 
				gan.save(network_dir + '/' + str(step) + '.h5')			


if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1], sys.argv[2])
