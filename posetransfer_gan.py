import tensorflow as tf
import os
import numpy as np
import sys
from keras import backend as K
import cv2
import datareader
import preprocess
import networks
import scipy.io as sio
from keras.callbacks import History
from keras.models import load_model,Model

n_joints = 14
batch_size = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
training_iters = 500000
gpu = '/gpu:3'
test_interval = 100
save_interval = 5000

#Dataset specific vars
n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'
n_train_examples = 2000
n_test_examples = batch_size


def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.readExamples_transfer(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples)

	queue_train = tf.train.slice_input_producer(ex_train,shuffle=True)
	queue_test = tf.train.slice_input_producer(ex_test,shuffle=True)

	param = {}
	stride = 4
	param['crop_size_x'] = 256
	param['crop_size_y'] = 256
	param['target_dist'] = 1.171
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['stride'] = stride
	param['sigma'] = 7


	with tf.device(gpu):
		generator = networks.network1(n_joints,IMG_HEIGHT,IMG_WIDTH,3,stride)
		discriminator = networks.discriminator(n_joints,IMG_HEIGHT,IMG_WIDTH,3,stride)
		gan = networks.gan(generator,discriminator,n_joints,IMG_HEIGHT,IMG_WIDTH,3,stride)

	'''
	generator.summary()
	discriminator.summary()
	gan.summary()
	'''

	hist = History()
	with tf.Session(config=config) as sess:
	
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		X_img = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))
		X_pose = np.zeros((batch_size, IMG_HEIGHT/stride,IMG_WIDTH/stride,n_joints*2))
		X_tgt = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))

		step = 0
    
		while step  < training_iters:
			for i in xrange(batch_size):
				ex = sess.run(queue_train)		
				I0,I1,H0,H1,_ = preprocess.makeInputOutputPair(ex,param)
				X_img[i,:,:,:] = I0
				X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)
				X_tgt[i,:,:,:] = I1

						
			#1. Get generated examples
			with tf.device(gpu):
				X_gen = generator.predict([X_img,X_pose], batch_size=batch_size)


			#2. Train Discriminator
			networks.make_trainable(discriminator,True)

			X_img_disc = np.concatenate((X_img,X_gen))
			X_pose_disc = np.concatenate((X_pose[:,:,:,0:n_joints],X_pose[:,:,:,n_joints:]))
			y1 = np.zeros([2*batch_size,2])
			y1[0:batch_size,1] = 1
			y1[batch_size:,0] = 1

			with tf.device(gpu):
				d_loss = discriminator.train_on_batch([X_img_disc, X_pose_disc],y1)

			networks.make_trainable(discriminator,False)
			
			for i in xrange(batch_size):
				ex = sess.run(queue_train)		
				I0,I1,H0,H1,_ = preprocess.makeInputOutputPair(ex,param)
				X_img[i,:,:,:] = I0
				X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)
				X_tgt[i,:,:,:] = I1

			y2 = np.zeros([batch_size,2])
			y2[:,1] = 1

			g_loss = gan.train_on_batch([X_img,X_pose],[X_tgt,y2])

			'''
			if(step % test_interval == 0):

				n_batches = batch_size/batch_size
				test_loss = 0
			
				for j in xrange(n_batches):
					for i in xrange(batch_size):
						ex = sess.run(queue_test)
				
						I0,I1,H0,H1,_,Mhalf = preprocess.makeInputOutputPair(ex,param)
						X_img[i,:,:,:] = I0
						X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)
						#X_mask[i,:,:,:] = M
						X_ctr[i,:,:,:] = Mhalf
						Y[i,:,:,:] = I1
					
					predy,pred_pose = model.predict([X_img,X_pose,X_ctr],batch_size=batch_size,verbose=0)
					d = predy - Y
					test_loss += np.sum(d**2)/(batch_size)
	
				test_loss /= n_batches
				print "1," + str(test_loss/(IMG_HEIGHT*IMG_WIDTH*3))
				sys.stdout.flush()

				sio.savemat('../results/outputs/network_pose/' + str(step) + '.mat',
                {'X_img': X_img,'Y': Y, 'Ypred': predy})	
	
			if(step % save_interval==0 and step > 0):
				model.save('../results/networks/network_pose/' + str(step) + '.h5')			
		
			'''

			step += 1	
		
		coord.request_stop()
		coord.join(threads)
		sess.close()


if __name__ == "__main__":
	train()

