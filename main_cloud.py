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
from keras.optimizers import Adam


from post_process import *
from masks import *
from DataGenerator import *
from dice_coef import *
from UNetlike import *
from CloudImage import cloudImage
from catalogueImage import catalogueImage

df_train = pd.read_csv("train.csv")
   
df_train['ImageID']          =   df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId']         =   df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence']   = ~ df_train['EncodedPixels'].isna()

df_train.head()


# Importation submission et test img
sub_df = pd.read_csv('sample_submission.csv')
sub_df['ImageID'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageID'].unique(), columns=['ImageID'])
print(test_imgs)


## Séparation des variables

df_train2 = df_train.iloc[:3000, :]

mask_count_df = df_train.groupby('ImageID').agg(np.sum).reset_index()
mask_count_df.sort_values('PatternPresence', ascending=False, inplace=True)
print(mask_count_df.index)

"""
## Encodage OneHot des classes
# http://www.kaggle.com/saneryee/understanding-clouds-keras-unet

train_ohe_df = df_train2[~ df_train2['EncodedPixels'].isnull()]
classes = train_ohe_df['PatternId'].unique()
train_ohe_df = train_ohe_df.groupby('ImageID')['PatternId'].agg(set).reset_index()

for class_name in classes:
    train_ohe_df[class_name] = train_ohe_df['PatternId'].map(lambda x: 1 if class_name in x else 0)
print(train_ohe_df.shape)
print(train_ohe_df.head())

# dictionary for fast access to ohe vectors
# key: ImageId
# value: ohe value
# {'0011165.jpg': array([1, 1, 0, 0]),
# '002be4f.jpg': array([1, 1, 1, 0]),...}
img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageID'], train_ohe_df.iloc[:, 2:].values)}
"""

### Séparation statifiée des variables train et val

#trat = train_ohe_df["PatternId"].map(lambda x: str(sorted(list(x))))
#print(strat)

train_idx, val_idx = train_test_split(mask_count_df.index, random_state=123, test_size=0.2) #ratify=strat

train_generator = DataGenerator(
    train_idx, 
    df=mask_count_df,
    target_df=df_train,
    reshape=(320, 480),
    augment=True,
    n_channels=3,
    n_classes=4)

val_generator = DataGenerator(
    val_idx, 
    df=mask_count_df,
    target_df= df_train, 
    reshape=(320, 480),
    augment=False,
    n_channels=3,
    n_classes=4)


## Définition du modèle UNet like
### TESTER AVEC UNET, RESNET

img_size=(320,480,3)
num_classes = 4

model = get_model(img_size, num_classes)
model.summary()

## Compilation  

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])
# model.compile(optimizer=Adam(lr = 3e-4), loss="binary_crossentropy", metrics=[dice_coef])
# Test avec binary_crossentropy, sparse_categorical_entropy

#### ENTRAINEMENT ####

history = model.fit_generator(
     train_generator,
     validation_data=val_generator,
     epochs=5,
     verbose=1)

# fonction de perte et dice_coeff
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(history.history["loss"], label = "dice loss")
plt.plot(history.history["val_loss"], label = "val dice loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
plt.legend()

plt.subplot(122)
plt.plot(history.history["dice_coef"], label="dice coef", color="red")
plt.plot(history.history["val_dice_coef"], label="val_dice coef", color="green")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("dice coef")
plt.show();



#### PREDICTION ####



### Stockage des pixels encodés dans sub_df 

best_threshold = 0.45
best_size = 15000

threshold = best_threshold
min_size = best_size

test_df = []
encoded_pixels = []
TEST_BATCH_SIZE = 500

for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
    batch_idx = list(range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE)))

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        gamma=0.8,
        base_path='./test_images/',
        target_df=sub_df,
        batch_size=1,
        n_classes=4)

    batch_pred_masks = model.predict_generator(test_generator,
                                               workers=1,
                                               verbose=1)

    # Predict out put shape is (320X480X4)
    # 4  = 4 classes, Fish, Flower, Gravel Surger.

    for j, idx in enumerate(batch_idx):
        filename = test_imgs['ImageID'].iloc[idx]
        image_df = sub_df[sub_df['ImageID'] == filename].copy()

        # Batch prediction result set
        pred_masks = batch_pred_masks[j, ]

        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[...,k].astype('float32')

            if pred_mask.shape != (350, 525):
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            pred_mask, num_predict = post_process(pred_mask, threshold, min_size)

            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)

sub_df['EncodedPixels'] = encoded_pixels


## Export des résultats en csv
sub_df.to_csv("sample_submission_test1.csv")

## Faire appraître les NaN
sub_df = pd.read_csv("test1_sample_submission.csv")

sub_df['FileName']          =   sub_df['Image_Label'].apply(lambda col: col.split('_')[0])
sub_df['PatternId']         =   sub_df['Image_Label'].apply(lambda col: col.split('_')[1])
sub_df['PatternPresence']   = ~ sub_df['EncodedPixels'].isna()

sub_df.head()


### Observation des résultats
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
plt.figure(figsize=[60, 20])
for index, row in sub_df[:8].iterrows():
    img = cv2.imread("./test_images/%s" % row['ImageID'])[...,[2, 1, 0]]
    img = cv2.resize(img, (525, 350))
    mask_rle = row['EncodedPixels']
    try: # label might not be there!
        mask = rle_decode(mask_rle)
    except:
        mask = np.zeros((1400, 2100))
    plt.subplot(2, 4, index+1)
    plt.imshow(img)
    plt.imshow(rle2mask(mask_rle, img.shape), alpha=0.5, cmap='gray')
    plt.title("Image %s" % (row['Image_Label']), fontsize=18)
    plt.axis('off')     
plt.show();

sns.set_style("white")
plt.figure(figsize=[60, 20])
for index, row in df_train[:8].iterrows():
    img = cv2.imread("./train_images/%s" % row['ImageID'])[...,[2, 1, 0]]
    img = cv2.resize(img, (525, 350))
    mask_rle = row['EncodedPixels']
    try: # label might not be there!
        mask = rle_decode(mask_rle)
    except:
        mask = np.zeros((1400, 2100))
    plt.subplot(2, 4, index+1)
    plt.imshow(img)
    plt.imshow(rle2mask(mask_rle, img.shape), alpha=0.5, cmap='gray')
    plt.title("Image %s" % (row['Image_Label']), fontsize=18)
    plt.axis('off')     
plt.show();

## OU
c = catalogueImage(dataFrame=sub_df, indexes = range(0,5), path="./test_images/")
c.visualizeCatalogue()

# visualisation masques entraînement
df_train["FileName"] = df_train["ImageID"].astype("str")
c = catalogueImage(path="./train_images/", indexes = range(0,5), dataFrame = df_train )
c.visualizeCatalogue()
