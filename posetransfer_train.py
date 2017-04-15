import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import preprocess
import networks
import scipy.io as sio
from keras.models import load_model,Model
from keras.optimizers import Adam

batch_size = 8
gpu = '/gpu:3'
test_interval = 200
save_interval = 5000

param = {}
param['IMG_HEIGHT'] = 256
param['IMG_WIDTH'] = 256
param['target_dist'] = 1.171
param['scale_max'] = 1.05
param['scale_min'] = 0.95
param['posemap_downsample'] = 4
param['sigma'] = 7
param['n_joints'] = 14

n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'
n_train_examples = 100000
n_test_examples = 1000

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.makeTransferExampleList(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples)

	train_feed = preprocess.transferExampleGenerator(ex_train,batch_size,param)
	test_feed = preprocess.transferExampleGenerator(ex_test,batch_size,param)

	with tf.device(gpu):
		model = networks.network1_weights(param)
		#model.compile(optimizer=Adam(lr=1e-4),loss='mse')

	with tf.Session(config=config) as sess:

		step = 0	
		while(True):
			X_img,X_pose,X_tgt,W = next(train_feed)

			#sio.savemat('test.mat', {'X_tgt': X_tgt, 'W': W})
			#return

			with tf.device(gpu):
				train_loss = model.train_on_batch([X_img,X_pose,W],X_tgt)
				train_loss /= (np.sum(W) * batch_size)

			print "0," + str(train_loss)
			sys.stdout.flush()
	
			if(step % test_interval == 0):
				n_batches = 8
	
				test_loss = 0		
				for j in xrange(n_batches):	
					X_img,X_pose,X_tgt,W = next(test_feed)
					pred_val = model.predict([X_img,X_pose,W],batch_size=batch_size)
					test_loss += np.sum((pred_val-X_tgt)**2)/(batch_size)
	
				test_loss /= (n_batches*param['IMG_HEIGHT']*param['IMG_WIDTH']*3)
				print "1," + str(test_loss)
				sys.stdout.flush()

				sio.savemat('../results/outputs/network1_weights/' + str(step) + '.mat',
         		{'X_img': X_img,'Y': X_tgt, 'Ypred': pred_val})	
	
			if(step % save_interval==0 and step > 0):
				model.save('../results/networks/network1_weights/' + str(step) + '.h5')			

			step += 1	


if __name__ == "__main__":
	train()

