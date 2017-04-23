def getParam():
	param = {}
	dn = 2 #downsample images
	param['IMG_HEIGHT'] = 256/dn 
	param['IMG_WIDTH'] = 256/dn
	param['obj_scale_factor'] = 1.171*200.0/dn
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['max_rotate_degree'] = 10
	param['max_sat_factor'] = 0.05
	param['max_px_shift'] = 10/dn
	param['posemap_downsample'] = 4/dn 
	param['sigma_joint'] = 7/dn
	param['n_joints'] = 14

	return param
