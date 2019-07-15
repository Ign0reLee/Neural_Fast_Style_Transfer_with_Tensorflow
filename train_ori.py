import os,sys
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_vgg.vgg19 as vgg
import time
from net.utils import *
from net.model_ori import *
from net.net_ori import *
from net.model_batch import *


content_path = Path_Load('./train')
style_path = './images/styles/composition_vii.jpg'

test_content_path ='./images/content/brad_pitt.jpg'
test_content_img = image_resize([test_content_path])
test_style_img = image_resize([style_path])


n_epoch = 2
batch_size = 1
lambda_s = 1e1
lambda_f =1e0
lambda_tv = 1e-2
learning_rate = 1e-3


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True



with tf.Session(config = config) as sess:
    model = Model(sess, lambda_s, lambda_f, lambda_tv, learning_rate)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    for epoch in range(n_epoch):
        batch = next_batch(style_path, content_path, batch_size)
        
        for i, b in enumerate(batch):
            st_time = time.time()
            
            st, ct = b
            loss, _ = model.train(st, ct)
            ed_time = time.time()
            print("Epoch : ", epoch, " Batch : ", i, " Loss : ", loss, " Time : ", ed_time - st_time)
            
            if i % 500 ==0:
                
                
                output = model.pred(test_style_img, test_content_img)
                r,g,b = cv2.split(output[0])
                output = cv2.merge([b,g,r])
                saver.save(sess, './Model_ori/'+style_path.split('/')[-1]+'_'+str(epoch)+'_'+str(i)+'.ckpt')
                cv2.imwrite('./result_ori/'+style_path.split('/')[-1]+'_'+str(epoch)+'_'+str(i)+'.jpg',output)
                
                
    output = model.pred(test_style_img, test_content_img)
    r,g,b = cv2.split(output[0])
    output = cv2.merge([b,g,r])
    saver.save(sess, './Model_ori/'+style_path.split('/')[-1]+'_'+str(epoch)+'_'+str(i)+'.ckpt')
    cv2.imwrite('./result_ori/'+style_path.split('/')[-1]+'_'+str(epoch)+'_'+str(i)+'.jpg',output)
                
