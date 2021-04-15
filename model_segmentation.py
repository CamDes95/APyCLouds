# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Generic function for testing different segmentation models (saving log and model .h5/.hdf5)
Possibility to tests different learning rate, freezing of not encoder weights, ...
Check existing images and masks in directories dir_image and dir_mask with same size than vector reduced_size
@author: luc eglin, camille desjardin, toufik saddik
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import dataGeneratorFromClass
import model_UNet
from bce_dice_loss import dice_coef

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
import segmentation_models as sm
from segmentation_models import get_preprocessing
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from segmentation_models.utils import set_trainable
from segmentation_models.losses import bce_dice_loss

# On modifie la précision flottante de 32-bits à 16-bits
"""from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')"""

"""
GPU OPTIONS
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

"""
Extract informations from dataframe
"""
df_train = pd.read_csv("train.csv")             # data frame of images and encoded pixels for masks
df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())
name_images = df_train['FileName'].unique()     # set of file names
n_images = len(name_images)                     # total number of indexed images


"""
Model and data parameters
"""
reduced_size = [224, 224]  # height x width     # size of input images (reduced)
backbones = ['resnet50','efficientnetb2']       # one or more models from segmentation_models library
backbones = ['resnet34','efficientnetb0','densenet121']
backbones = ['efficientnetb2']
#backbones = ['resnet50']
encoder_weights='imagenet'                      # initialization of the encoder weights
l_rates = [1e-04,5e-04,1e-03,5e-03,1e-02]       # one or more learning rate for optimization
l_rates = [0.001]
n_train = 4500                                  # number of training images
n_valid = .2*n_train                            # 
max_valid = int(min(n_train+n_valid, n_images)) # number of validation images
dir_image="reduced_train_images_224/"           # directory of input images
dir_mask="reduced_train_masks_224/"             # directory of input masks
n_epochs = 11                                   # number of epochs for fit
batch_size = 4                                  # batch size
dataAugmentation = True                         # boolean for data augmentation
encoder_freeze = True                           # boolean for encoder weights freezing

for BACKBONE in backbones:
    """
    Build model from base model 
    """
    preprocess_input = get_preprocessing(BACKBONE)
    base_model = sm.Unet(BACKBONE, 
                 classes=4,
                 encoder_weights=encoder_weights,
                 encoder_freeze=encoder_freeze)
    
    # gray level image: 1 canal
    inp = Input(shape=(None, None, 1))
    
    """
    Because we have graylevel images as input: we need that to adapt to the models from segmentation_models
    """
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    
    out = base_model(l1)    
    """out = Activation('softmax', dtype='float32', name='predictions')(out) """
    
    for lr in l_rates:
        print("\nModel " + BACKBONE + ", learning rate " + str(lr) + "\n")
        
        model = Model(inp, out, name=base_model.name)
        
        """
        # RMS optimization/ 
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop"
        )"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr
        )

    
        model.compile(optimizer=optimizer,
                      loss=bce_dice_loss,
                      metrics=dice_coef)
        
        # Datagenerator for training set
        data_gen = dataGeneratorFromClass.DataGeneratorFromClass(list_IDs=np.arange(n_train),
                                        list_images=name_images,
                                        dim=reduced_size,
                                        batch_size=batch_size,
                                        dir_image=dir_image,
                                        dir_mask=dir_mask,
                                        augment=dataAugmentation)
        # Datagenerator for validation set
        val_gen = dataGeneratorFromClass.DataGeneratorFromClass(list_IDs=np.arange(n_train, max_valid),
                                        list_images=name_images,
                                        dim=reduced_size,
                                        batch_size=batch_size,
                                        dir_image=dir_image,
                                        dir_mask=dir_mask)
        
        # Callback to stop if NaN for loss function
        TON = callbacks.TerminateOnNaN()
        
        # Record model when the score is better than before
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #"checkpoints/UNet_{epoch:02d}_{val_dice_coef:.2f}.h5", 
        "checkpoints/"+'model0.4_'+BACKBONE+"_encoder_weights_"+encoder_weights+'_lr'+str(optimizer.learning_rate.numpy())+'_'+str(n_epochs)+'epochs_'+ ''.join(['' if (data_gen.augment==True) else 'No'])+'DataAugment'+ ''.join(['' if (encoder_freeze==True) else 'No'])+'EncoderFreeze_{epoch:02d}_{val_dice_coef:.2f}.h5',
        monitor='val_dice_coef', 
        verbose=1, 
        mode = 'max', 
        save_weights_only=False, 
        save_best_only=True
        )

        # Callback for changing learning rate according to the epoch
        def decreasinglrUpdate(epoch, learning_rate):
            if epoch % 15 == 0 & epoch>0:
                return learning_rate * 0.5
            else:
                return learning_rate
    
    
        lrScheduler = callbacks.LearningRateScheduler(schedule=decreasinglrUpdate, verbose=1)
    
        # Log file of the dice coeff and loss function, according to the epochs
        csv_logger = CSVLogger('log0.4_'+BACKBONE+"_encoder_weights_"+encoder_weights+'_lr'+str(optimizer.learning_rate.numpy())+'_'+str(n_epochs)+'epochs_'+ ''.join(['' if (data_gen.augment==True) else 'No'])+'DataAugment'+'.csv', append=False, separator=';')
        # Stop if results do not improve
        callback_earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', 
                                                    mode='max', 
                                                    patience=5,
                                                    verbose=1)
        
        """
        # run this part if fine tuning...
        history = model.fit_generator(data_gen, 
                                      epochs=15, 
                                      callbacks=[TON, callback_earlyStopping, csv_logger,checkpoint], 
                                      validation_data=val_gen)
        
        
        for l in model.layers[2].layers:
            l.trainable = True"""
            
        """optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr/2
        )
        model.compile(optimizer=optimizer,
                      loss=bce_dice_loss,
                      metrics=dice_coef)"""
        
        # Fitting model
        history = model.fit_generator(data_gen, 
                                      epochs=n_epochs, 
                                      callbacks=[TON, callback_earlyStopping, csv_logger,checkpoint], 
                                      validation_data=val_gen)
        
        # Save final model
        model.save('model0.4_'+BACKBONE+"_encoder_weights_"+encoder_weights+'_lr'+str(optimizer.learning_rate.numpy())+'_'+str(n_epochs)+'epochs_'+ ''.join(['' if (data_gen.augment==True) else 'No'])+'DataAugment'+ ''.join(['' if (encoder_freeze==True) else 'No'])+'EncoderFreeze'+'.hdf5')
    # Free model
    del model