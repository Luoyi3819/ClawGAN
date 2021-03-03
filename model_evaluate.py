from __future__ import print_function, division
import scipy

import tensorflow.compat.v1 as tf #use version 1.0
tf.disable_v2_behavior() #
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Drvisout, Concatenate
from keras.layers import BatchNormalization, Activation, Zervisadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.vistimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import scipy.misc

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model,load_model
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img
import fid
from sewar.full_ref import uqi,psnr,ssim,vifp,mse,rmse,msssim,psnrb  #不同的指标

img_res = (256,256)
path_inf = os.path.join('test_data/infrared')  #320x240
inf_list=os.listdir(path_inf)
inf_list.sort()
path_vis=os.path.join('test_data/visible')  #320x240
vis_list=os.listdir(path_vis)
vis_list.sort()
lenth = len(vis_list)

BA_model_path = 'result/models/generator200.h5'
BA_model=load_model(BA_model_path, custom_objects = {"InstanceNormalization": InstanceNormalization}, compile=False)

totol_metric_dict = {"mse":0.0,"rmse":0.0,"uqi":0.0,"ssim":0.0,"psnr":0.0,"psnrb":0.0,"vifp":0.0, "fid":0.0}  #metric

imgs_inf = []
imgs_vis = []
for (inf_path,vis_path) in zip(inf_list, vis_list):
    img_inf = img_to_array(load_img(os.path.join(path_inf,inf_path),color_mode='rgb'))
    img_vis = img_to_array(load_img(os.path.join(path_vis,vis_path),color_mode='rgb'))


    img_inf = resize(img_inf, img_res)
    img_vis = resize(img_vis, img_res)
    
    imgs_inf = [img_inf]
    imgs_vis = [img_vis]

    imgs_inf = np.array(imgs_inf)/127.5 - 1.
    imgs_vis = np.array(imgs_vis)/127.5 - 1.
    
    #A for visible, B for infrared
    fake_A = BA_model.predict(imgs_inf)       #fake image by Generator
    #fake_B = AB_model.predict(imgs_vis)
    #reconstr_A = BA_model.predict(fake_B)
    #reconstr_B = AB_model.predict(fake_A)
    fake_A = fake_A*0.5 + 0.5
    #fake_B = fake_B*0.5 + 0.5
    #reconstr_A = reconstr_A*0.5 + 0.5
    #reconstr_B = reconstr_B*0.5 + 0.5
    #fake_A = (fake_A + 1) * 127.5   #fake visible and img_vis(real_visible)

    #save

    os.makedirs("test_data/fake_visible", exist_ok = True)
    cv2.imwrite("test_data/fake_visible/" + vis_path , fake_A[0][:,:,::-1] * 255)

    totol_metric_dict_matched = {"mse":0.0,"rmse":0.0,"uqi":0.0,"ssim":0.0,"psnr":0.0,"psnrb":0.0,"vifp":0.0}  #参数指标

true_path = "test_data/visible"
fake_path = "test_data/fake_visiblee"

lenth = len(os.listdir(true_path))

for true_name,fake_name in zip(os.listdir(true_path),os.listdir(fake_path)):
	true = cv2.imread(os.path.join(true_path,true_name))
	fake = cv2.imread(os.path.join(fake_path,fake_name))

	metric_dict_matched = {"mse":mse(fake,true),"rmse":rmse(fake,true),"uqi":uqi(fake,true),"ssim":ssim(fake,true)[0] \
	   				,"psnr":psnr(fake,true),"psnrb":psnrb(fake,true),"vifp":vifp(fake,true)}
	for key,value in metric_dict_matched.items():
		totol_metric_dict_matched[key] = totol_metric_dict_matched[key]+value

for key,value in totol_metric_dict_matched.items():
	totol_metric_dict_matched[key] /= lenth
print(totol_metric_dict_matched)
#path = ["train_data/" + method + "_infrared","train_data/" + method + "_visible"]
path = [true_path,fake_path]
fid_value = fid.calculate_fid_given_paths(path, inception_path = None, low_profile=False)
print("FID: ", fid_value)  
print("done")
