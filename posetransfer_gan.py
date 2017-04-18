import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import preprocess
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
from keras.optimizers import Adam

batch_size = 8
gpu = '/gpu:0'
test_interval = 200
model_save_interval = 5000
test_save_interval=200

n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'
n_train_examples = 100000
n_test_examples = 1000

params = param.getParam()
n_joints = params['n_joints']

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.makeTransferExampleList(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples)

	train_feed = preprocess.transferExampleGenerator(ex_train,batch_size,params)
	test_feed = preprocess.transferExampleGenerator(ex_test,batch_size,params)
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			generator = networks.network1(params)
			discriminator = networks.discriminator(params)
			discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5))
			gan = networks.gan(generator,discriminator,params)
			gan.compile(optimizer=Adam(lr=1e-4),loss=['mse','categorical_crossentropy'],loss_weights=[20.0,1.0])
	
		step = 0	
		while(True):
			X_img,X_pose,X_tgt = next(train_feed)			
	
			#Get generator output
			with tf.device(gpu):
				X_gen = generator.predict([X_img,X_pose])
	
			#Train discriminator
			networks.make_trainable(discriminator,True)
			X_img_disc = np.concatenate((X_img,X_gen))
			X_pose_disc = np.concatenate((X_pose[:,:,:,0:n_joints],X_pose[:,:,:,n_joints:]))
			y = np.zeros([2*batch_size,2])
			y[0:batch_size,1] = 1
			y[batch_size:,0] = 1		
			with tf.device(gpu):
				d_loss = discriminator.train_on_batch([X_img_disc, X_pose_disc],y)

			networks.make_trainable(discriminator,False)

			#TRAIN GAN
			X_img,X_pose,X_tgt = next(train_feed)			
			y = np.zeros([batch_size,2])
			y[:,1] = 1
			gan_loss = gan.train_on_batch([X_img,X_pose],[X_tgt,y])
			print "0," + str(gan_loss[0]) + "," + str(gan_loss[1]) + "," + str(gan_loss[2])
			sys.stdout.flush()

			if(step % test_interval == 0):
				n_batches = 8
				test_loss = np.array([0.0,0.0,0.0])			
				for j in xrange(n_batches):	
					X_img,X_pose,X_tgt = next(test_feed)	
					y = np.zeros([batch_size,2])
					y[:,0] = 1
					test_loss_j = gan.test_on_batch([X_img,X_pose], [X_tgt,y])
					test_loss += np.array(test_loss_j)
	
				test_loss /= (n_batches) #*param['IMG_HEIGHT']*param['IMG_WIDTH']*3)
				print "1," + str(test_loss[0]) + "," + str(test_loss[1]) + "," + str(test_loss[2])
				sys.stdout.flush()


			if(step % test_save_interval==0):
				X_img,X_pose,X_tgt = next(test_feed)
				pred_val = gan.predict([X_img,X_pose])
				X_pred = pred_val[0]
		
				sio.savemat('../results/outputs/network_gan/' + str(step) + '.mat',
         		{'X_img': X_img,'X_tgt': X_tgt, 'pred': X_pred})	

	
			if(step % model_save_interval==0): 
				gan.save('../results/networks/network_gan/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	train()

