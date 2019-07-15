import os,sys
import tensorflow as tf
import tensorflow_vgg.vgg19 as vgg


def conv2d(X, Out, Kernel, Stride, Padding='SAME', Name=None):
    with tf.variable_scope(Name):
        return tf.layers.conv2d(X, Out, Kernel, Stride, padding=Padding)


def Wconv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    # set convolution layers.
    assert isinstance(x, tf.Tensor)
    return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)

def upconv2d(X, Out, Kernel, Stride, Padding='SAME', Name=None):
    with tf.variable_scope(Name):
        return tf.layers.conv2d_transpose(X, Out, Kernel, Stride, Padding)    


def ReLU(X):
    return tf.nn.relu(X)


def fc_layer(X, Out, Name=None):
    with tf.variable_scope(Name):
        return tf.layers.dense(X, Out)


def batch_norm(X):
    
    epsilon = 1e-5
    
    mean, variance = tf.nn.moments(X, axes=[0, 1, 2])

    norm_batch = tf.nn.batch_normalization(X, mean, variance, 0, 1, epsilon)

    return norm_batch


def Residual_block(X, Out, Kernel,Stride, Name=None):
    
    h = ReLU(batch_norm(conv2d(X, Out, Kernel, Stride, 'SAME', Name+'_conv_1')))
    h = batch_norm(conv2d(h, Out, Kernel, Stride, 'SAME', Name+'_conv_2'))
    
    return X + h


def gram_matrix(x):
    
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [-1, h*w, ch])
    # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
    return gram

def total_variation_regularization(x, beta=1):
    
    assert isinstance(x, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: Wconv2d(x, wh, p='SAME')
    tvw = lambda x: Wconv2d(x, ww, p='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    
    return tv


