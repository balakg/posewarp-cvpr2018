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

	ex_train,ex_test = datareader.makeTransferExampleList(params)

	train_feed = datageneration.transferExampleGenerator(ex_train,params)
	test_feed = datageneration.transferExampleGenerator(ex_test,params)

	batch_size = params['batch_size']

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = VGG19(weights='imagenet',include_top=False,
								input_shape=(128,128,3))
			networks.make_trainable(vgg_model,False)
			generator = networks.network_warp(params,vgg_model)
			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
			gan = networks.gan(generator,discriminator,params)
			gan.compile(optimizer=Adam(lr=1e-4),loss=['mse','mse','binary_crossentropy'],
						loss_weights=[1.0,0.001,0.001])

		step = 0	
		while(True):
			X_src,X_tgt,X_pose,X_mask,X_trans = next(train_feed)			
	
			#Get generator output
			with tf.device(gpu):
				inputs = [X_src,X_pose,X_mask,X_trans]
				X_gen = generator.predict(inputs)[0]	
	
			#Train discriminator
			networks.make_trainable(discriminator,True)

			#Make a batch of batch_size generated examples and batch_size 
			#real examples. Each example is a source/target pair of images and poses.
			X_src_disc = np.concatenate((X_src[:,:,:,0:3],X_src[:,:,:,0:3]))
			X_tgt_disc = np.concatenate((X_gen,X_tgt))
			X_pose_disc = np.concatenate((X_pose,X_pose))

			y = np.zeros([2*batch_size,2])
			y[0:batch_size,1] = 1
			y[batch_size:,0] = 1

			with tf.device(gpu):
				inputs = [X_src_disc,X_tgt_disc,X_pose_disc]
				d_loss = discriminator.train_on_batch(inputs,y)

			networks.make_trainable(discriminator,False)

			#TRAIN GAN
			X_src,X_tgt,X_pose,X_mask,X_trans = next(train_feed)			
			y = np.zeros([batch_size,2])
			y[:,0] = 1 #Pretend these are real.

			inputs = [X_src,X_pose,X_mask,X_trans]
			X_feat = vgg_model.predict(util.vgg_preprocess(X_tgt))
			gan_loss = gan.train_on_batch(inputs,[X_tgt,X_feat,y])
			util.printProgress(step,0,gan_loss)


			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(4)			
				for j in xrange(n_batches):	
					X_src,X_tgt,X_pose,X_mask,X_trans = next(test_feed)			
					y = np.zeros([batch_size,2])
					y[:,1] = 1 #Fake images

					inputs = [X_src,X_pose,X_mask,X_trans]
					X_feat = vgg_model.predict(util.vgg_preprocess(X_tgt))
					test_loss_j = gan.test_on_batch(inputs, [X_tgt,X_feat,y])
					test_loss += np.array(test_loss_j)
	
				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			if(step % params['test_save_interval']==0):
				X_src,X_tgt,X_pose,X_mask,X_trans = next(test_feed)			
				inputs = [X_src,X_pose,X_mask,X_trans]
				pred_val = gan.predict(inputs)
	
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X_src': X_src,'X_tgt': X_tgt, 'pred': pred_val[0]}) 

			
			if(step % params['model_save_interval']==0): 
				gan.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train('golfswinghd', sys.argv[1], sys.argv[2])
