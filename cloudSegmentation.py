import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import dataGeneratorFromClass
import model_UNet
from bce_dice_loss import bce_dice_loss, dice_loss, dice_coef
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import callbacks
#import visuPredict

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

reduced_size = [144, 224]  # height x width
reduced_size = [352, 528]  # height x width

df_train = pd.read_csv("train.csv")

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

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)

model.compile(optimizer=optimizer,
              # optimizer='adam',
              loss=bce_dice_loss,
              #loss='categorical_crossentropy',
              metrics=dice_coef)

n_train = 4800
n_valid = .2*n_train
max_valid = int(min(n_train+n_valid, n_images))

data_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(n_train),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16,
                                dir_image="reduced_train_images_3/",
                                dir_mask="reduced_train_masks_3/")
val_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(n_train, max_valid),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16,
                                dir_image="reduced_train_images_3/",
                                dir_mask="reduced_train_masks_3/")

TON = callbacks.TerminateOnNaN()


def decreasinglrUpdate(epoch, learning_rate):
    if epoch % 3 == 0 & epoch>0:
        return learning_rate * 0.1
    else:
        return learning_rate


lrScheduler = callbacks.LearningRateScheduler(schedule=decreasinglrUpdate,
                                              verbose=1)

# Permet d'enregistrer les modèles tout au long de l'entrainement
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/UNet_{epoch:02d}_{val_dice_coef:.2f}.h5", monitor='val_dice_coef', verbose=1, mode = 'max', save_weights_only=True, save_best_only=False
)

history = model.fit_generator(data_gen, epochs=5, callbacks=[TON, lrScheduler, checkpoint], validation_data=val_gen) # , validation_data=val_gen) #, callbacks=callbacks)

model.save('model_UNet_2.hdf5')

