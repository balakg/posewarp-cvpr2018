from keras.applications.vgg19 import VGG19
from keras.layers import AveragePooling2D, Conv2D, Input
from keras.models import Model

# Our VGG implementation has the following differences from the standard vgg model:
# 1. input size is now 256x256
# 2. average pooling instead of max pooling to remove some grid-like artifacts during synthesis

def vgg_norm():
    img_input = Input(shape=(256, 256, 3))
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)
    x3 = AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x2)

    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x4)
    x6 = AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x5)

    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x6)
    x8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x7)
    x9 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x8)
    x10 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x9)
    x11 = AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x10)

    x12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x11)
    x13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x12)
    x14 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x13)
    x15 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x14)
    x16 = AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x15)

    x17 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x16)
    x18 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x17)
    x19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x18)
    x20 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x19)
    x21 = AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x20)

    model = Model(inputs=[img_input], outputs=[x1, x2, x4, x5, x7, x8, x9, x10, x12, x13, x14, x15])
    model_orig = VGG19(weights='imagenet', input_shape=(256, 256, 3), include_top=False)

    for i in range(len(model.layers)):
        weights = model_orig.layers[i].get_weights()
        model.layers[i].set_weights(weights)

    return model
