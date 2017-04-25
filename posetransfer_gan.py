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
from keras.models import load_model,Model
from keras.optimizers import Adam

batch_size = 8
gpu = '/gpu:3'
test_interval = 100
test_save_interval=200
model_save_interval = 5000

n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
img_sfx = '.jpg'
n_train_examples = 100000
n_test_examples = 1000

params = param.getParam()
n_joints = params['n_joints']
model_name = 'gan'

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.makeTransferExampleList(
		vid_pth,info_pth,n_test_vids,5,img_sfx,n_train_examples,n_test_examples)

	train_feed = datageneration.transferExampleGenerator(ex_train,batch_size,params)
	test_feed = datageneration.transferExampleGenerator(ex_test,batch_size,params)
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			generator = networks.network_warp(params)
			discriminator = networks.discriminator(params)
			discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
			gan = networks.gan(generator,discriminator,params)
			gan.compile(optimizer=Adam(lr=1e-4),loss=['mse','binary_crossentropy'],loss_weights=[10.0,1.0])

	
		step = 0	
		while(True):
			X_src,X_tgt,X_pose,X_mask = next(train_feed)			
	
			#Get generator output
			with tf.device(gpu):
				X_gen = generator.predict([X_src,X_pose,X_mask])
	
			#Train discriminator
			networks.make_trainable(discriminator,True)

			X_src_disc = np.concatenate((X_src[:,:,:,0:3],X_src[:,:,:,0:3]))
			X_tgt_disc = np.concatenate((X_gen,X_tgt))
			X_pose_disc = np.concatenate((X_pose,X_pose))

			y = np.zeros([2*batch_size,2])
			y[0:batch_size,1] = 1
			y[batch_size:,0] = 1
		
			with tf.device(gpu):
				d_loss = discriminator.train_on_batch([X_src_disc, X_tgt_disc, X_pose_disc],y)

			networks.make_trainable(discriminator,False)

			#TRAIN GAN
			X_src,X_tgt,X_pose,X_mask = next(train_feed)			
			y = np.zeros([batch_size,2])
			y[:,0] = 1

			gan_loss = gan.train_on_batch([X_src,X_pose,X_mask],[X_tgt,y])
			print str(step) + ",0," + str(gan_loss[0]) + "," + str(gan_loss[1]) + "," + str(gan_loss[2])
			sys.stdout.flush()

			if(step % test_interval == 0):
				n_batches = 8
				test_loss = np.array([0.0,0.0,0.0])			
				for j in xrange(n_batches):	
					X_src,X_tgt,X_pose,X_mask = next(test_feed)			
					y = np.zeros([batch_size,2])
					y[:,0] = 1
					test_loss_j = gan.test_on_batch([X_src,X_pose,X_mask], [X_tgt,y])
					test_loss += np.array(test_loss_j)
	
				test_loss /= (n_batches)
				print str(step) + ",1," + str(test_loss[0]) + "," + str(test_loss[1]) + "," + str(test_loss[2])
				sys.stdout.flush()


			if(step % test_save_interval==0):
				X_src,X_tgt,X_pose,X_mask = next(test_feed)			
				pred_val = gan.predict([X_src,X_pose,X_mask])
		
				sio.savemat('../results/outputs/' + model_name + '/' + str(step) + '.mat',
         		{'X_src': X_src,'X_tgt': X_tgt, 'pred': pred_val[0]})	

			
			if(step % model_save_interval==0): 
				gan.save('../results/networks/' + model_name + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	train()

