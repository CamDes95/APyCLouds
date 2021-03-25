import pandas as pd
import numpy as np
import cloudImage
import os
from time import time
from bce_dice_loss import bce_dice_loss, dice_coef
import tensorflow as tf
from tensorflow.keras import callbacks
from model_EfficientNet import EfficientNet_model
import numpy as np
import tensorflow.keras
import cloudImage
import dataGeneratorFromClass
import tensorflow as tf

#import visuPredict

"""
Objectif 2eme itération :
    
    utiliser transfer learning EfficientNet
    adapter le learning rate pour meilleurs résultats
    augmentation des données dans datagenerator
    activation sigmoid ou softmax a tester xsur output


""" 

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)

"""d = dataGenerator.DataGenerator(list_IDs=[0], list_images=name_images, dim=reduced_size, batch_size=1)
X, y = d.__getitem__(0)"""


reduced_size = (160,224)
input_shape = (160, 224, 3)
img_h = 160
img_w = 224
n_channels = 3
n_classes = 4

model = EfficientNet_model(img_h, img_w, n_channels, n_classes = 4)

"""
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")
"""

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  ### lr : 1e-3 1e-4 1e-2

model.compile(optimizer=optimizer,
              loss=bce_dice_loss,
              metrics=[dice_coef])


data_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(1000),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16)
val_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(1200,1400),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16)

TON = callbacks.TerminateOnNaN()

'''
def decreasinglrUpdate(epoch, learning_rate):
    if epoch % 3 == 0:
        return learning_rate * 0.1
    else:
        return learning_rate


lrScheduler = callbacks.LearningRateScheduler(schedule=decreasinglrUpdate,
                                              verbose=1)
'''


history = model.fit(data_gen,
                    epochs=5,
                    callbacks=[TON],
                    validation_data=val_gen) #, callbacks=callbacks)

print(history.history['accuracy'])

model.save('model_EfficientNet-1.h5')
