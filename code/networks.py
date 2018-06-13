import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Input, UpSampling2D
from keras.layers import concatenate, Flatten, Reshape, Lambda
from keras.layers import LeakyReLU, MaxPooling2D
import keras


def my_conv(x_in, nf, ks=3, strides=1, activation='lrelu', name=None):
    x_out = Conv2D(nf, kernel_size=ks, padding='same', strides=strides)(x_in)

    if activation == 'lrelu':
        x_out = LeakyReLU(0.2, name=name)(x_out)
    elif activation != 'none':
        x_out = Activation(activation, name=name)(x_out)

    return x_out


def vgg_loss(feat_net, feat_weights, n_layers, reg=0.1):
    def loss_fcn(y_true, y_pred):
        y_true_feat = feat_net(Lambda(vgg_preprocess)(y_true))
        y_pred_feat = feat_net(Lambda(vgg_preprocess)(y_pred))

        loss = []
        for j in range(n_layers):

            std = feat_weights[str(j)][1] + reg
            std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std, 0), 0), 0)
            d = tf.subtract(y_true_feat[j], y_pred_feat[j])
            loss_j = tf.reduce_mean(tf.abs(tf.divide(d, std)))

            if j == 0:
                loss = loss_j
            else:
                loss = tf.add(loss, loss_j)
        return loss / (n_layers * 1.0)

    return loss_fcn


def vgg_preprocess(arg):
    z = 255.0 * (arg + 1.0) / 2.0
    r = z[:, :, :, 0] - 103.939
    g = z[:, :, :, 1] - 116.779
    b = z[:, :, :, 2] - 123.68
    return tf.stack([r, g, b], axis=3)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def discriminator(param):
    img_h = param['IMG_HEIGHT']
    img_w = param['IMG_WIDTH']
    n_joints = param['n_joints']
    pose_dn = param['posemap_downsample']

    x_tgt = Input(shape=(img_h, img_w, 3))
    x_src_pose = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    x_tgt_pose = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))

    x = my_conv(x_tgt, 64, ks=5)
    x = MaxPooling2D()(x) # 128
    x = concatenate([x, x_src_pose, x_tgt_pose])
    x = my_conv(x, 128, ks=5)
    x = MaxPooling2D()(x) # 64
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 32
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 16
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 8
    x = my_conv(x, 256)  # 8

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[x_tgt, x_src_pose, x_tgt_pose], outputs=y, name='discriminator')
    return model


def wass(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def gan(gen_model, disc_model, param):

    img_h = param['IMG_HEIGHT']
    img_w = param['IMG_WIDTH']
    n_joints = param['n_joints']
    n_limbs = param['n_limbs']
    pose_dn = param['posemap_downsample']

    src_in = Input(shape=(img_h, img_w, 3))
    pose_src = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    pose_tgt = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    mask_in = Input(shape=(img_h, img_w, n_limbs+1))
    trans_in = Input(shape=(2, 3, n_limbs+1))

    make_trainable(disc_model, False)
    y_gen = gen_model([src_in, pose_src, pose_tgt, mask_in, trans_in])
    y_class = disc_model([y_gen, pose_src, pose_tgt])

    gan_model = Model(inputs=[src_in, pose_src, pose_tgt, mask_in, trans_in],
                      outputs=[y_gen, y_class], name='gan')

    return gan_model


def repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, dtype='int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


def meshgrid(height, width):
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                    tf.cast(width, tf.float32) - 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                    tf.cast(height, tf.float32) - 1.0, height), 1),
                                    tf.ones(shape=tf.stack([1, width])))
    return x_t, y_t


def interpolate(inputs):
    im = inputs[0]
    x = inputs[1]
    y = inputs[2]

    im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")

    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, 'float32') + 1
    y = tf.cast(y, 'float32') + 1

    max_x = tf.cast(width - 1, 'int32')
    max_y = tf.cast(height - 1, 'int32')

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    dim2 = width
    dim1 = width * height
    base = repeat(tf.range(num_batch) * dim1, (height - 2) * (width - 2))

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2

    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.cast(im_flat, 'float32')

    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x1_f = tf.cast(x1, 'float32')
    y1_f = tf.cast(y1, 'float32')

    dx = x1_f - x
    dy = y1_f - y

    wa = tf.expand_dims((dx * dy), 1)
    wb = tf.expand_dims((dx * (1 - dy)), 1)
    wc = tf.expand_dims(((1 - dx) * dy), 1)
    wd = tf.expand_dims(((1 - dx) * (1 - dy)), 1)

    output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    output = tf.reshape(output, tf.stack([-1, height - 2, width - 2, channels]))
    return output


def affine_warp(im, theta):
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]

    x_t, y_t = meshgrid(height, width)
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
    grid = tf.expand_dims(grid, 0)
    grid = tf.reshape(grid, [-1])
    grid = tf.tile(grid, tf.stack([num_batch]))
    grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

    T_g = tf.matmul(theta, grid)
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    x_s_flat = tf.reshape(x_s, [-1])
    y_s_flat = tf.reshape(y_s, [-1])

    return interpolate([im, x_s_flat, y_s_flat])


def make_warped_stack(args):
    mask = args[0]
    src_in = args[1]
    trans_in = args[2]

    for i in range(11):
        mask_i = K.repeat_elements(tf.expand_dims(mask[:, :, :, i], 3), 3, 3)
        src_masked = tf.multiply(mask_i, src_in)

        if i == 0:
            warps = src_masked
        else:
            warp_i = affine_warp(src_masked, trans_in[:, :, :, i])
            warps = tf.concat([warps, warp_i], 3)

    return warps


def unet(x_in, pose_in, nf_enc, nf_dec):
    x0 = my_conv(x_in, nf_enc[0], ks=7)  # 256
    x1 = my_conv(x0, nf_enc[1], strides=2)  # 128
    x2 = concatenate([x1, pose_in])
    x3 = my_conv(x2, nf_enc[2])
    x4 = my_conv(x3, nf_enc[3], strides=2)  # 64
    x5 = my_conv(x4, nf_enc[4])
    x6 = my_conv(x5, nf_enc[5], strides=2)  # 32
    x7 = my_conv(x6, nf_enc[6])
    x8 = my_conv(x7, nf_enc[7], strides=2)  # 16
    x9 = my_conv(x8, nf_enc[8])
    x10 = my_conv(x9, nf_enc[9], strides=2)  # 8
    x = my_conv(x10, nf_enc[10])

    skips = [x9, x7, x5, x3, x0]
    for i in range(5):
        x = UpSampling2D()(x)
        x = concatenate([x, skips[i]])
        x = my_conv(x, nf_dec[i])

    return x


def network_posewarp(param):
    img_h = param['IMG_HEIGHT']
    img_w = param['IMG_WIDTH']
    n_joints = param['n_joints']
    pose_dn = param['posemap_downsample']
    n_limbs = param['n_limbs']

    # Inputs
    src_in = Input(shape=(img_h, img_w, 3))
    pose_src = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    pose_tgt = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    src_mask_prior = Input(shape=(img_h, img_w, n_limbs+1))
    trans_in = Input(shape=(2, 3, n_limbs+1))

    # 1. FG/BG separation
    x = unet(src_in, pose_src, [64]*2 + [128]*9, [128]*4 + [32])
    src_mask_delta = my_conv(x, 11, activation='linear')
    src_mask = keras.layers.add([src_mask_delta, src_mask_prior])
    src_mask = Activation('softmax', name='mask_src')(src_mask)

    # 2. Separate into fg limbs and background
    warped_stack = Lambda(make_warped_stack)([src_mask, src_in, trans_in])
    fg_stack = Lambda(lambda arg: arg[:, :, :, 3:], output_shape=(img_h, img_w, 3*n_limbs),
                      name='fg_stack')(warped_stack)
    bg_src = Lambda(lambda arg: arg[:, :, :, 0:3], output_shape=(img_h, img_w, 3),
                    name='bg_src')(warped_stack)
    bg_src_mask = Lambda(lambda arg: tf.expand_dims(arg[:, :, :, 0], 3))(src_mask)

    # 3. BG/FG synthesis
    x = unet(concatenate([bg_src, bg_src_mask]), pose_src, [64]*2 + [128]*9, [128]*4 + [64])
    bg_tgt = my_conv(x, 3, activation='tanh', name='bg_tgt')

    # x = unet(fg_stack, pose_tgt, [64]*2 + [128]*9, [128]*4 + [64])
    x = unet(fg_stack, pose_tgt, [64] + [128] * 3 + [256] * 7, [256, 256, 256, 128, 64])

    fg_tgt = my_conv(x, 3, activation='tanh', name='fg_tgt')

    fg_mask = my_conv(x, 1, activation='sigmoid', name='fg_mask_tgt')
    fg_mask = concatenate([fg_mask, fg_mask, fg_mask])
    bg_mask = Lambda(lambda arg: 1 - arg)(fg_mask)

    # 5. Merge bg and fg
    fg_tgt = keras.layers.multiply([fg_tgt, fg_mask], name='fg_tgt_masked')
    bg_tgt = keras.layers.multiply([bg_tgt, bg_mask], name='bg_tgt_masked')
    y = keras.layers.add([fg_tgt, bg_tgt])

    model = Model(inputs=[src_in, pose_src, pose_tgt, src_mask_prior, trans_in], outputs=[y])

    return model


def network_unet(param):
    n_joints = param['n_joints']
    pose_dn = param['posemap_downsample']
    img_h = param['IMG_HEIGHT']
    img_w = param['IMG_WIDTH']

    src_in = Input(shape=(img_h, img_w, 3))
    pose_src = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    pose_tgt = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))

    x = unet(src_in, concatenate([pose_src, pose_tgt]), [64] + [128] * 3 + [256] * 7,
             [256, 256, 256, 128, 64])
    y = my_conv(x, 3, activation='tanh')

    model = Model(inputs=[src_in, pose_src, pose_tgt], outputs=[y])
    return model
