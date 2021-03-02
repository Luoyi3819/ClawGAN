from __future__ import print_function, division
import scipy

import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.disable_v2_behavior() #禁用2.0版本的方法
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
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
import fid
from keras.models import Model,load_model
#from sewar.full_ref import uqi, psnr, ssim, vifp, mse, rmse, msssim, psnrb  #不同的指标
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img


img_res = (256,256)
path_sar = os.path.join('test_data/TM_CCORR_NORMED_infrared')  #320x240
sar_list=os.listdir(path_sar)
sar_list.sort()
path_op=os.path.join('test_data/TM_CCORR_NORMED_visible')  #320x240
op_list=os.listdir(path_op)
op_list.sort()
lenth = len(op_list)
BA_model_path = 'cycle_loss_result/models/generator180.h5'
#AB_model_path = 'cycle_Unet++_loss_result/models/generatorAB100.h5'
BA_model=load_model(BA_model_path, custom_objects = {"InstanceNormalization": InstanceNormalization}, compile=False)
#AB_model=load_model(AB_model_path, custom_objects = {"InstanceNormalization": InstanceNormalization})  #加载自定义的函数或层
totol_metric_dict = {"mse":0.0,"rmse":0.0,"uqi":0.0,"ssim":0.0,"psnr":0.0,"psnrb":0.0,"vifp":0.0, "fid":0.0}  #参数指标

imgs_sar = []
imgs_op = []
for (sar_path,op_path) in zip(sar_list, op_list):
    img_sar = img_to_array(load_img(os.path.join(path_sar,sar_path),color_mode='rgb'))
    img_op = img_to_array(load_img(os.path.join(path_op,op_path),color_mode='rgb'))


    img_sar = resize(img_sar, img_res)
    img_op = resize(img_op, img_res)
    
    imgs_sar = [img_sar]
    imgs_op = [img_op]

    imgs_sar = np.array(imgs_sar)/127.5 - 1.
    imgs_op = np.array(imgs_op)/127.5 - 1.
    
    #A 是visible B 是infrared
    fake_A = BA_model.predict(imgs_sar)
    #fake_B = AB_model.predict(imgs_op)
    #reconstr_A = BA_model.predict(fake_B)
    #reconstr_B = AB_model.predict(fake_A)
    fake_A = fake_A*0.5 + 0.5
    #fake_B = fake_B*0.5 + 0.5
    #reconstr_A = reconstr_A*0.5 + 0.5
    #reconstr_B = reconstr_B*0.5 + 0.5
    #fake_A = (fake_A + 1) * 127.5   #fake visible and img_op(real_visible)

    #保存生成图片

    os.makedirs("test_data/cycle_loss_fake_visible", exist_ok = True)
    cv2.imwrite("test_data/cycle_loss_fake_visible/" + op_path , fake_A[0][:,:,::-1] * 255)
    # os.makedirs("test_data/fake_infrared", exist_ok = True)
    # cv2.imwrite("test_data/fake_infrared/" + sar_path , fake_B[0][:,:,::-1] * 255)
    # os.makedirs("test_data/reconstr_visible", exist_ok = True)
    # cv2.imwrite("test_data/reconstr_visible/" + op_path , reconstr_A[0][:,:,::-1] * 255)
    # os.makedirs("test_data/reconstr_infrared", exist_ok = True)
    # cv2.imwrite("test_data/reconstr_infrared/" + op_path , reconstr_B[0][:,:,::-1] * 255)

    #cv2和matplot通道顺序不一样
    #plt.axis("off")
    # plt.imshow(fake_A[0])
    # plt.savefig(os.path.join("test_data/fake_visible",op_path)， pad_inches=0.0) 

    # metric_dict = {"mse":mse(fake_A[0],img_op),"rmse":rmse(fake_A[0],img_op),"uqi":uqi(fake_A[0],img_op)}  #参数指标
    # ,
    #         "ssim":ssim(fake_A[0],img_op)[0], "psnr":psnr(fake_A[0],img_op),"psnrb":psnrb(fake_A[0],img_op),
    #         "vifp":vifp(fake_A[0],img_op)


#     for key, value in metric_dict.items():
# 		    totol_metric_dict[key] = totol_metric_dict[key] + value

# for key, value in totol_metric_dict.items():
#     totol_metric_dict[key] /= lenth

# path = ['test_data/visible','test_data/fake_visible']
# totol_metric_dict["fid"] = fid.calculate_fid_given_paths(path, inception_path = None, low_profile=False)
# print(totol_metric_dict)
print("done")



