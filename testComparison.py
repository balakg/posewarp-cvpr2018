import tensorflow as tf
import os
import numpy as np
import sys
import datareader
import datageneration
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
from keras.optimizers import Adam
import h5py
import util
import myVGG
import skimage
from skimage.measure import compare_ssim

def l1Error(pred,true):
	return np.mean(np.abs(pred-true))

def ssimError(pred,true):

	true = (true+1.0)/2.0
	true = true.astype('float32')
	pred = (pred+1.0)/2.0

	ssim = 0
	for i in xrange(pred.shape[0]):
		ssim += compare_ssim(pred[i,:,:,:],true[i,:,:,:],multichannel=True)

	return (ssim/(1.0*pred.shape[0]))

def vggError(pred_vgg,true_vgg,feat_weights):
	loss = 0
	for j in xrange(12):
		std = feat_weights[str(j)][1]
		std = np.expand_dims(np.expand_dims(np.expand_dims(std,0),0),0)
		d = true_vgg[j] - pred_vgg[j]
		loss += np.mean(np.abs(d/std))

	return loss/12.0


def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')
	aux_params = param.getDatasetParams('test-aux')

	_,lift_test = datareader.makeWarpExampleList(lift_params,0,2000,2,1)
	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)
	_,workout_test = datareader.makeWarpExampleList(workout_params,0,2000,2,3)
	_,tennis_test = datareader.makeWarpExampleList(tennis_params,0,2000,2,4)
	_,aux_test = datareader.makeWarpExampleList(aux_params,0,2000,2,5)

	test = lift_test + golf_test+workout_test + tennis_test + aux_test
	feed = datageneration.warpExampleGenerator(test,params,do_augment=False,return_pose_vectors=True)
	

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = myVGG.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')

			fgbg = networks.network_fgbg(params,vgg_model,response_weights)
			fgbg.load_weights('../results/networks/fgbg/170000.h5')	

			gen = networks.network_fgbg(params,vgg_model,response_weights)
			disc = networks.discriminator(params)
			gan = networks.gan(gen,disc,params,vgg_model,response_weights,0.1,1e-4)
			gan.load_weights('../results/networks/wgan/1000.h5')
			
			fgbg_l1 = networks.network_fgbg(params,vgg_model,response_weights,loss='l1')
			fgbg_l1.load_weights('../results/networks/fgbg_l1/88000.h5')	

	
		np.random.seed(17)
		n_batches = 50
		#metrics = np.zeros(9)
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)		

			#pred_vgg = fgbg.predict(X[:-2])
			#vgg_error = vggError(pred_vgg_feat,Y_feat,response_weights)

			pred_l1 = fgbg_l1.predict(X[:-2])
			pred_vgg = fgbg.predict(X[:-2])
			pred_gan = gen.predict(X[:-2])

			'''
			l1_feat = vgg_model.predict(util.vgg_preprocess(pred_l1))
			vgg_feat = vgg_model.predict(util.vgg_preprocess(pred_vgg)) 
			gan_feat = vgg_model.predict(util.vgg_preprocess(pred_gan))
			Y_feat = vgg_model.predict(util.vgg_preprocess(Y))

			preds = [pred_l1,pred_vgg,pred_gan]
			vgg_preds = [l1_feat,vgg_feat,gan_feat]
	
			l1_metrics = [l1Error(preds[i],Y) for i in xrange(len(preds))]
			vgg_metrics = [vggError(vgg_preds[i],Y_feat,response_weights) for i in xrange(len(vgg_preds))]
			ssim_metrics = [ssimError(preds[i],Y) for i in xrange(len(preds))]	

			metrics += np.array(l1_metrics+vgg_metrics+ssim_metrics)
			print metrics/(1.0*(j+1))	
			'''
			sio.savemat('results/comparison_fgbg/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred_vgg': pred_vgg, 'pred_l1': pred_l1, 
						'pred_gan': pred_gan,'src_pose': X[-2],'tgt_pose': X[-1]})
		
		#metrics /= (1.0*n_batches)
		#print metrics

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
