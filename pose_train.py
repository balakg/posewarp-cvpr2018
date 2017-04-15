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
from keras.models import load_model

n_joints = 14
batch_size = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
training_iters = 500000
gpu = '/gpu:0'
test_interval = 500
save_interval = 5000

#Dataset specific vars
n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'


def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.readExamples_pose(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx)

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


	#model = networks.posePredictor(n_joints,IMG_HEIGHT,IMG_WIDTH,3,4)
	model = load_model('../poseresults/networks/network1/30000.h5')

	with tf.Session(config=config) as sess:
	
		#sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		X = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))
		X_ctr = np.zeros((batch_size,IMG_HEIGHT/stride,IMG_WIDTH/stride,1))
		Y = np.zeros((batch_size, IMG_HEIGHT/stride,IMG_WIDTH/stride,n_joints))

		step = 30001
    
		while step  < training_iters:
			for i in xrange(batch_size):
				ex = sess.run(queue_train)		
				I,H,M = preprocess.makeInputOutputPair_pose(ex,param)
				X[i,:,:,:] = I
				X_ctr[i,:,:,:] = M
				Y[i,:,:,:] = H

			with tf.device(gpu):
				train_loss = model.train_on_batch([X,X_ctr],Y)

			print "0," + str(train_loss)
			sys.stdout.flush()


			if(step % test_interval == 0):

				n_batches = 500/batch_size
				test_loss = 0
			
				for j in xrange(n_batches):
					for i in xrange(batch_size):
						ex = sess.run(queue_test)		
						I,H,M = preprocess.makeInputOutputPair_pose(ex,param)
						X[i,:,:,:] = I
						X_ctr[i,:,:,:] = M
						Y[i,:,:,:] = H
	
					pred_val = model.predict([X,X_ctr],batch_size=batch_size,verbose=0)
					d = pred_val - Y
					test_loss += np.sum(d**2)/(batch_size)

				test_loss /= n_batches
				print "1," + str(test_loss)
				sys.stdout.flush()

				sio.savemat('../poseresults/outputs/network1/' + str(step) + '.mat',
                {'X': X,'Y': Y, 'Ypred': pred_val})	
	
			if(step % save_interval==0 and step > 0):
				model.save('../poseresults/networks/network1/' + str(step) + '.h5')			


			step += 1	
		
		coord.request_stop()
		coord.join(threads)
		sess.close()

if __name__ == "__main__":
	train()
