#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


###### ETAPES #######

# Définir une classe DataGenerator pour charger les imgs et mask au fur et a mesure de l'entraînement
    # (important : init et méthode get_item = générer sample pr train (retourne x et y (image et img à prédire)))
    
    
# Définir la fonction de perte dice_loss
# Création et compilation du modèle (UNet, EfficientNet, denseNet... => algo de segmentation)
# entraîner modèle sur données d'entraînement et données générées

# Transfer Learning avec efficientNet ?
# Callbacks??
# PR AUC Mean svt use pour évaluer les perfs du modèle
# Itération = MAJ des param du modèle


# In[ ]:





# In[ ]:


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


# In[3]:


get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
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

import numpy as np
import pandas as pd


# In[4]:


import pandas as pd
import numpy as np

df_train = pd.read_csv("train.csv")
   
df_train['ImageID']          =   df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId']         =   df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence']   = ~ df_train['EncodedPixels'].isna()

df_train.iloc[:1000, :]


# In[ ]:





# In[ ]:





# In[6]:


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# In[7]:


#  https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))
    
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks

def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)
        
        rle = mask2rle(mask)
        rles.append(rle)
        
    return rles


# In[8]:


# Datagenerator de Ying : https://www.kaggle.com/gogo827jz/resunet-keras-with-some-new-ideas

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='./train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None, gamma=None,
                 augment=False, n_classes=4, random_state=2021, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.gamma = gamma
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            
            if self.augment:
                X, y = self.__augment_batch(X, y)
            
            return X, y
        
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageID'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)
            
            if self.reshape is not None:
                img = np_resize(img, self.reshape)
            
            # Adjust gamma
            if self.gamma is not None:
                img = adjust_gamma(img, gamma=self.gamma)
            
            # Store samples
            X[i,] = img

        return X
    
    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df["ImageID"].iloc[ID]
            image_df = self.target_df[self.target_df["ImageID"] == im_name]
            
            rles = image_df['EncodedPixels'].values
            
            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)
            
            y[i, ] = masks

        return y
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img
    
    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1)
        ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    
    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])
        
        return img_batch, masks_batch


# In[9]:


# Définition fonction de perte
#  https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# In[10]:


# 1000 premieres lignes pour tester
df_train2 = df_train.iloc[:200, :]

mask_count_df = df_train2.groupby('ImageID').agg(np.sum).reset_index()
mask_count_df.sort_values('PatternPresence', ascending=False, inplace=True)
print(mask_count_df.index)


# In[24]:


print(mask_count_df)


# In[13]:


BATCH_SIZE = 12

train_idx, val_idx = train_test_split(mask_count_df.index, random_state=2019, test_size=0.2)

train_generator = DataGenerator(
    train_idx, 
    df=mask_count_df,
    target_df=df_train,
    batch_size=BATCH_SIZE,
    reshape=(320, 480),
    gamma=0.8,
    augment=True,
    n_channels=3,
    n_classes=4
)

val_generator = DataGenerator(
    val_idx, 
    df=mask_count_df,
    target_df= df_train,
    batch_size= BATCH_SIZE, 
    reshape=(320, 480),
    gamma=0.8,
    augment=False,
    n_channels=3,
    n_classes=4
)


# In[23]:


from tensorflow.keras import layers

# AUTOENCODER

img_size=(320,480)
num_classes = 4


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()


# In[25]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size=12

history = model.fit_generator(
     train_generator,
     validation_data=val_generator,
     steps_per_epoch = 100,
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


# In[31]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




