# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Data segmentation with UNet model
@author: luc eglin, camille desjardin, toufik saddik
"""

import pandas as pd
import numpy as np
import dataGeneratorFromClass
import model_UNet
from bce_dice_loss import bce_dice_loss, dice_loss, dice_coef
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
#from tensorflow import keras
from tensorflow.keras import callbacks
#import visuPredict

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

reduced_size = [224, 224]  # height x width

df_train = pd.read_csv("train.csv")
l_rates = [1e-05,1e-04,1e-03,1e-02,1e-01,1]


df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)

"""d = dataGenerator.DataGenerator(list_IDs=[0], list_images=name_images, dim=reduced_size, batch_size=1)
X, y = d.__getitem__(0)"""

model=model_UNet.multi_unet_model(n_classes=4,
                                  IMG_HEIGHT=reduced_size[0],
                                  IMG_WIDTH=reduced_size[1],
                                  IMG_CHANNELS=1)

for lr in l_rates:
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=lr,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        
        name="RMSprop"
    )
    #optimizer = tf.keras.optimizers.Adam(
    #    learning_rate=1e-5
    #)
    
    model.compile(optimizer=optimizer,
                  # optimizer='adam',
                  loss=bce_dice_loss,
                  #loss='categorical_crossentropy',
                  metrics=dice_coef)
    
    n_train = 4600
    n_valid = .2*n_train
    max_valid = int(min(n_train+n_valid, n_images))
    
    data_gen = dataGeneratorFromClass.DataGeneratorFromClass(list_IDs=np.arange(n_train),
                                    list_images=name_images,
                                    dim=reduced_size,
                                    batch_size=16,
                                    dir_image="reduced_train_images_224/",
                                    dir_mask="reduced_train_masks_224/",
                                    augment=True)
    val_gen = dataGeneratorFromClass.DataGeneratorFromClass(list_IDs=np.arange(n_train, max_valid),
                                    list_images=name_images,
                                    dim=reduced_size,
                                    batch_size=16,
                                    dir_image="reduced_train_images_224/",
                                    dir_mask="reduced_train_masks_224/")
    
    TON = callbacks.TerminateOnNaN()
    
    csv_logger = CSVLogger('log_aug_withoutweightsInit_50epochs_UNet_lr_'+str(optimizer.learning_rate.numpy())+'.csv', append=False, separator=';')
    
    history = model.fit_generator(data_gen, epochs=50, callbacks=[TON, csv_logger], validation_data=val_gen) # , validation_data=val_gen) #, callbacks=callbacks)
    model.save('model_aug_withoutweightsInit_UNet_lr_'+str(optimizer.learning_rate.numpy())+'.hdf5')

