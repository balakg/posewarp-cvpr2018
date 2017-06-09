def getGeneralParams():
	param = {}
	dn = 1 #downsample images from 256 to 128 for now
	param['IMG_HEIGHT'] = 256/dn 
	param['IMG_WIDTH'] = 256/dn
	param['obj_scale_factor'] = 1.14/dn
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['max_rotate_degree'] = 5
	param['max_sat_factor'] = 0.05
	param['max_px_shift'] = 5
	param['posemap_downsample'] = 2
	param['sigma_joint'] = 7/4.0
	param['n_joints'] = 14

	param['test_interval'] = 100
	param['test_save_interval'] = 1000
	param['model_save_interval'] = 5000
	param['project_dir'] = '/afs/csail.mit.edu/u/b/balakg/pose/pose2image'

	param['batch_size'] = 4
	param['seq_len'] = 2
	return param

def getDatasetParams(dataset):
	param = {}

	if(dataset == 'golfswinghd'):
		param['n_test_vids'] = 13
		param['vid_pth'] = '../../datasets/golfswinghd/videos/'
		param['info_pth'] = '../../datasets/golfswinghd/videoinfo/'
		param['img_sfx'] = '.jpg'
		param['test_vids'] = None

	if(dataset == 'weightlifting'):
		param['n_test_vids'] = 6
		param['vid_pth'] = '../../datasets/weightlifting/videos/Men/'
		param['info_pth'] = '../../datasets/weightlifting/videoinfo/'
		param['img_sfx'] = '.png'
		param['test_vids'] = [1,7,18,29,33,57]

	if(dataset == 'pennaction'):
		param['vid_pth'] = '../../datasets/Penn_Action/frames'
		param['info_pth'] = '../../datasets/Penn_Action/labels'
		param['img_sfx'] = '.jpg'

	if(dataset == 'yoga'):
		param['vid_pth'] = '../../datasets/yoga/frames'
		param['info_pth'] = '../../datasets/yoga/videoinfo'
		param['img_sfx'] = '.jpg'
		param['n_test_vids'] = 5
		param['test_vids'] = None

	if(dataset == 'famous'):
		param['vid_pth'] = '../../datasets/famous/frames'
		param['info_pth'] = '../../datasets/famous/info'
		param['img_sfx'] = '.jpg'
		param['n_test_vids'] = 1
		param['test_vids'] = None
	return param
