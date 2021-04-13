import pandas as pd
import numpy as np
import cloudImage
import os
from time import time
from bce_dice_loss import bce_dice_loss, dice_coef
import tensorflow as tf
from tensorflow.keras import callbacks
from model_EfficientNetB0 import EfficientNet_model
from model_EfficientNetB2 import EfficientNetB2_model
from model_EfficientNetB4 import EfficientNetB4_model
import numpy as np
import tensorflow.keras
import cloudImage
import dataGeneratorFromClass
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


#### CHARGEMENT ET MISE EN FORME DONNEES ####

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['Image'].unique()  # FileName en Image pour AUC ROC
n_images = len(name_images)

# Importation submission et test img
sub_df = pd.read_csv('sample_submission.csv')
sub_df['ImageID'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageID'].unique(), columns=['ImageID'])
print(test_imgs)

#name_images_test = sub_df["FileName"].unique()

"""d = dataGenerator.DataGenerator(list_IDs=[0], list_images=name_images, dim=reduced_size, batch_size=1)
X, y = d.__getitem__(0)"""


#### DEFINITION DU MODELE ####

reduced_size = (224,224)

img_h = 224
img_w = 224
n_channels = 3
n_classes = 4

model = EfficientNetB4_model(img_h, img_w, n_channels, n_classes = 4)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 

model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coef])


#### GENERATEUR DE DONNEES ####

data_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(4500),
                                list_images=name_images,
                                dim=reduced_size,
                                dir_image="reduced_train_images_224/",
                                batch_size=16)
val_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(4501,5501),
                                list_images=name_images,
                                dim=reduced_size,
                                dir_mask="reduced_train_masks_224/",
                                batch_size=16)
"""
test_gen = dataGeneratorFromClass.DataGenerator(list_IDs = np.arange(3000),
                                                list_images = name_images_test,
                                                dim = reduced_size,
                                                dir_image = "reduced_test_images_256",
                                                batch_size = 32)
"""
TON = callbacks.TerminateOnNaN()

#### ENTRAINEMENT ####

history = model.fit(data_gen,
                    epochs=5,
                    callbacks=[TON],
                    validation_data=val_gen) #, callbacks=callbacks)


# Visualisation loss et dice_coef lors de l'entraînement :
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(history.history["loss"], label = "dice loss")
plt.plot(history.history["val_loss"], label = "val dice loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
plt.legend()

plt.subplot(122)
plt.plot(history.history["dice_coef"], label="dice coef", color="red")
plt.plot(history.history["val_dice_coef"], label="val dice coef", color="green")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("dice coef")
plt.show();

# Sauvegarde du modèle et des poids
model.save('model_EffNetB0_4_TOP_imgnet.h5')
model.save_weights("model_weights_EffNetB2_4_TOP_imgnet.h5")


# Performance du modèle sur données test
