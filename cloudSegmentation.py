import pandas as pd
import numpy as np
import dataGeneratorFromClass
import model_UNet
from bce_dice_loss import bce_dice_loss, dice_loss
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import callbacks
#import visuPredict

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

reduced_size = [144, 224]  # height x width
reduced_size = [350, 525]  # height x width

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
              metrics=['accuracy'])

n_train = 4800
n_valid = .2*n_train
max_valid = int(min(n_train+n_valid, n_images))

data_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(n_train),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16,
                                dir_image="reduced_train_images_2/",
                                dir_mask="reduced_train_masks_2/")
val_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(n_train, max_valid),
                                list_images=name_images,
                                dim=reduced_size,
                                batch_size=16,
                                dir_image="reduced_train_images_2/",
                                dir_mask="reduced_train_masks_2/")

TON = callbacks.TerminateOnNaN()


def decreasinglrUpdate(epoch, learning_rate):
    if epoch % 3 == 0 & epoch>0:
        return learning_rate * 0.1
    else:
        return learning_rate


lrScheduler = callbacks.LearningRateScheduler(schedule=decreasinglrUpdate,
                                              verbose=1)

history = model.fit_generator(data_gen, epochs=10, callbacks=[TON, lrScheduler], validation_data=val_gen) # , validation_data=val_gen) #, callbacks=callbacks)
print(history.history['accuracy'])
model.save('model_UNet_aug_size.hdf5')
