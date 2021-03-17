"""
AutoEncoder : 
=> compresser img dans un petit vecteur (couches de convolution : SeparableCOnv)
convlution séparable = une image pr chq filtre séparement
1 neurone dans convol séparable va prédire une image
=> Trés efficace en terme de temps de calcul
=> décompression du vecteur pour obtenir l'image originale (Conv2DTranspose puis UpSampling)
Conv2Dtranspose = génére une grande matrice à partir d'une petite
=> si input = 2x2, kernel = 3,3 => output = 4x4
Upsampling = augmente la taille de la matrice (ex : matrice 2x2 devient 4x4)
===> Entraînement du modèle à prédire l'image d'origine (pas de classif !!!!)
     Enleve du bruit dans une image!
Aprés entraînement, on enléve la 2e moitiée (=décompression) et ajout couches classif
Bénéfices : 
- utile pour transfer learning
- enlève le bruit dans l'image
- utile pour la segmentation 
=> prédiction du masque de segmentation de l'image et vérification si correct ou pas
=> Prédiction de 4 masques, pas de masques= img noire
strides de 2 de Conv2D = pas des pixels de 2 au lieu de 1 
==> output 2x plus petit que img originale, si input = 160,160 output = 80,80
+ stride élevé et plus taille de output petit
Unet = AutoEncoder spécial
"""

import os
os.chdir("./Desktop/APyClouds/understanding_cloud_organization")

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import numpy as np
import pandas as pd
import os
import json

import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import albumentations as albu



from post_process import *
from masks import *
from DataGenerator import *
from dice_coef import *
from UNetlike import *


df_train = pd.read_csv("train.csv")
   
df_train['ImageID']          =   df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId']         =   df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence']   = ~ df_train['EncodedPixels'].isna()

df_train.head()





## Séparation des variables

# 200 images pour tester
df_train2 = df_train.iloc[:800, :]

mask_count_df = df_train2.groupby('ImageID').agg(np.sum).reset_index()
mask_count_df.sort_values('PatternPresence', ascending=False, inplace=True)
print(mask_count_df.index)

train_idx, val_idx = train_test_split(mask_count_df.index, random_state=2019, test_size=0.2)

train_generator = DataGenerator(
    train_idx, 
    df=mask_count_df,
    target_df=df_train,
    reshape=(320, 480),
    gamma=0.8,
    augment=True,
    n_channels=3,
    n_classes=4)

val_generator = DataGenerator(
    val_idx, 
    df=mask_count_df,
    target_df= df_train, 
    reshape=(320, 480),
    gamma=0.8,
    augment=False,
    n_channels=3,
    n_classes=4)


## Définition du modèle UNet like

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size=(320,480)
num_classes = 4

model = get_model(img_size, num_classes)
model.summary()

## Compilation avec 

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])


# In[ ]:


history = model.fit_generator(
     train_generator,
     validation_data=val_generator,
     steps_per_epoch = len(train_idx)//batch_size,
     validation_steps = len(train_idx)//batch_size,
     epochs=10)


# In[27]:


plt.subplot(121)
plt.plot(history.history["loss"])
plt.xlabel("epochs")
plt.ylabel("loss function")

plt.subplot(122)
plt.plot(history.history["acc"], label="acc")
plt.plot(history.history["val_acc"], label = "val_acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")


