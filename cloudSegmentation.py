# Enleve tous les messages de debuggage de tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
from bce_dice_loss import bce_dice_loss, dice_coef
from tensorflow import keras
from model_ResNet import ResNet_model
import dataGeneratorFromClass
import matplotlib.pyplot as plt


#### CHARGEMENT ET MISE EN FORME DONNEES ####

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)

# Importation submission et test img
sub_df = pd.read_csv('sample_submission.csv')
sub_df['ImageID'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageID'].unique(), columns=['ImageID'])
print(test_imgs)

#name_images_test = sub_df["FileName"].unique()

"""
d = dataGenerator.DataGenerator(list_IDs=[0], list_images=name_images, dim=reduced_size, batch_size=1)
X, y = d.__getitem__(0)
"""


#### DEFINITION DU MODELE ####

reduced_size = (224,224)
img_h = 224
img_w = 224
n_channels = 3
n_classes = 4

model = ResNet_model(img_h, img_w, n_channels, n_classes)


optimizer = keras.optimizers.Adam(learning_rate=1e-3) 

model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coef])


#### GENERATEUR DE DONNEES ####

data_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(1000),
                                list_images=name_images,
                                dim=reduced_size,
                                dir_image="reduced_train_images/",
                                batch_size=32)
val_gen = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(1100,1300),
                                list_images=name_images,
                                dim=reduced_size,
                                dir_mask="reduced_train_masks/",
                                batch_size=32)

TON = keras.callbacks.TerminateOnNaN()

#### ENTRAINEMENT ####

history = model.fit(data_gen,
                    epochs=5,
                    callbacks=[TON],
                    validation_data=val_gen)


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
plt.savefig('ResNet152V2_loss_4.png')
plt.show();

# Sauvegarde du modèle et des poids
model.save('model_ResNet152V2_4.h5')
model.save_weights("model_weights_ResNet152V2_4.h5")

