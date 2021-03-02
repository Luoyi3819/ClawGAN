from __future__ import print_function, division
import scipy

import cv2
import tensorflow.compat.v1 as tf #1.0 version
tf.disable_v2_behavior() #ban 2.0 version
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D ,Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from data_loader import DataLoader


class CycleGAN_Unetplus_loss():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        
        self.result_model_dir = "cycle_Unet++_loss_result/models/"
        self.result_image_dir = "cycle_Unet++_loss_result/images/"
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        self.lambda_syn = 20                        # Synthesized loss
        self.lambda_fr = 10                         #fake reconstr loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        # self.g_AB = load_model("cycle_Unet++_loss_result/models/generatorAB40.h5", custom_objects = {"InstanceNormalization": InstanceNormalization}) 
        # self.g_BA = load_model("cycle_Unet++_loss_result/models/generatorBA40.h5", custom_objects = {"InstanceNormalization": InstanceNormalization})

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],                                               
                              outputs=[ valid_A, valid_B,   #[valid, valid,
                                        fake_A, fake_B,       #imgs_A, imgs_B
                                        reconstr_A, reconstr_B,   #imgs_A, imgs_B,
                                        img_A_id, img_B_id,     #imgs_A, imgs_B,
                                        reconstr_A, reconstr_B ])  #fake_A, fake_B]) 
        self.combined.compile(loss=['mse', 'mse',             # MSE（L2 loss）and MAE（L1 loss）
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  10, 10,    
                                            self.lambda_syn, self.lambda_syn,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id,
                                            self.lambda_fr, self.lambda_fr ],
                            optimizer=optimizer)

    def build_generator(self):
        """unet_plus_plus Unet++ generator"""

        inputs = Input(shape = self.img_shape)
        conv0_0 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv0_0 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_0)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

        conv1_0 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv1_0 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_0)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

        up1_0 = Conv2DTranspose(self.gf, (2, 2), strides=(2, 2), padding='same')(conv1_0)
        merge00_10 = Concatenate()([conv0_0,up1_0])
        conv0_1 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00_10)
        conv0_1 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_1)

        conv2_0 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv2_0 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_0)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

        up2_0 = Conv2DTranspose(self.gf*2, (2, 2), strides=(2, 2), padding='same')(conv2_0)
        merge10_20 = Concatenate()([conv1_0,up2_0])
        conv1_1 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10_20)
        conv1_1 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)

        up1_1 = Conv2DTranspose(self.gf, (2, 2), strides=(2, 2), padding='same')(conv1_1)
        merge01_11 = Concatenate()([conv0_0,conv0_1,up1_1])
        conv0_2 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge01_11)
        conv0_2 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_2)

        conv3_0 = Conv2D(self.gf*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv3_0 = Conv2D(self.gf*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_0)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

        up3_0 = Conv2DTranspose(self.gf*4, (2, 2), strides=(2, 2), padding='same')(conv3_0)
        merge20_30 = Concatenate()([conv2_0,up3_0])
        conv2_1 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge20_30)
        conv2_1 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)

        up2_1 = Conv2DTranspose(self.gf*2, (2, 2), strides=(2, 2), padding='same')(conv2_1)
        merge11_21 = Concatenate()([conv1_0,conv1_1,up2_1])
        conv1_2 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11_21)
        conv1_2 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)

        up1_2 = Conv2DTranspose(self.gf, (2, 2), strides=(2, 2), padding='same')(conv1_2)
        merge02_12 = Concatenate()([conv0_0,conv0_1,conv0_2,up1_2])
        conv0_3 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge02_12)
        conv0_3 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_3)

        conv4_0 = Conv2D(self.gf*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv4_0 = Conv2D(self.gf*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_0)

        up4_0 = Conv2DTranspose(self.gf*8, (2, 2), strides=(2, 2), padding='same')(conv4_0)
        merge30_40 = Concatenate()([conv3_0,up4_0])
        conv3_1 = Conv2D(self.gf*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge30_40)
        conv3_1 = Conv2D(self.gf*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)

        up3_1 = Conv2DTranspose(self.gf*4, (2, 2), strides=(2, 2), padding='same')(conv3_1)
        merge21_31 = Concatenate()([conv2_0,conv2_1,up3_1])
        conv2_2 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge21_31)
        conv2_2 = Conv2D(self.gf*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_2)

        up2_2 = Conv2DTranspose(self.gf*2, (2, 2), strides=(2, 2), padding='same')(conv2_2)
        merge12_22 = Concatenate()([conv1_0,conv1_1,conv1_2,up2_2], )
        conv1_3 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge12_22)
        conv1_3 = Conv2D(self.gf*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_3)

        up1_3 = Conv2DTranspose(self.gf, (2, 2), strides=(2, 2), padding='same')(conv1_3)
        merge03_13 = Concatenate()([conv0_0,conv0_1,conv0_2,conv0_3,up1_3])
        conv0_4 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge03_13)
        conv0_4 = Conv2D(self.gf, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_4)
        # 二分类任务
        #conv0_4 = Conv2D(1, 1, activation = 'sigmoid')(conv0_4)
        output_img = Conv2D(self.channels, kernel_size=4, padding='same', activation='tanh')(conv0_4)

        #model = Model(input = inputs, output = conv0_4)
        
        return Model(inputs,output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                #imgs_B for infrared, imgs_A for visible
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators  
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B,
                                                        fake_A, fake_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            if epoch % 20 == 0:
                os.makedirs(self.result_model_dir, exist_ok=True)
                self.g_BA.save(self.result_model_dir + 'generatorBA%d.h5' % (epoch))
                self.g_AB.save(self.result_model_dir + 'generatorAB%d.h5' % (epoch))
    def sample_images(self, epoch, batch_i):
        os.makedirs(self.result_image_dir, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        imgs_A = 0.5 * imgs_A + 0.5
        fake_A = 0.5 * fake_A + 0.5
        reconstr_A = 0.5 * reconstr_A + 0.5
        #gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        gen_imgs = [imgs_A, fake_A, reconstr_A]
        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j][i])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.result_image_dir + "%d_%d.png" % (epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CycleGAN_Unetplus_loss()
    gan.train(epochs=200, batch_size=3, sample_interval=20)


