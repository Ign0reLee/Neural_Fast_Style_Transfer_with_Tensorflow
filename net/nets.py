import os,sys
import numpy as np
import tensorflow as tf
import tensorflow_vgg.vgg19 as vgg
from net.utils import *

def Network(X):
    
    kernels = [9 ,3, 3, 1, 1, 1, 1, 1, 3, 3, 9]
    strides = [1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1]
    channels = [32,64,128,128,128,128,128,128,64,32,3]
    h = X
    for i, kernel in enumerate(kernels):
        if i<3:
            h = batch_norm(ReLU(conv2d(h, channels[i], kernel, strides[i], 'SAME', 'Encode_conv_' + str(i))))
        elif i<8:
            h = Residual_block(h, channels[i], kernel, strides[i], 'Residual_block_' + str(i-3))
        elif i<10:
            h = batch_norm(ReLU(upconv2d(h,  channels[i], kernel, strides[i], 'SAME', 'Decode_conv_' + str(i-8))))
        else:
            h = upconv2d(h, channels[i], kernel, strides[i], 'SAME', 'Decode_conv_' + str(i-8))
            
    return  tf.div(tf.subtract(h,tf.reduce_min(h)),tf.subtract(tf.reduce_max(h),tf.reduce_min(h))) * 255.
