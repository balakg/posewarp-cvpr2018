def getGeneralParams():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = 256/dn
    param['IMG_WIDTH'] = 256/dn
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 14
    param['n_limbs'] = 10
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    param['test_interval'] = 500
    param['model_save_interval'] = 5000
    param['project_dir'] = '/afs/csail.mit.edu/u/b/balakg/pose/pose2image'

    param['batch_size'] = 4
    param['seq_len'] = 2
    return param

