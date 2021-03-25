# https://youtu.be/XyX5HNuv-xE
# https://youtu.be/q-p8v1Bxvac
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D,Dropout,  MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
import efficientnet.tfkeras as efn 
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB0
from tensorflow.keras import Sequential

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


### IMAGE AUGMENTATION A TESTER
img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

################################################################
input_shape = (160,224,3)
img_h = 160
img_w = 224
n_channels = 3
n_classes = 4

def EfficientNet_model(img_h, img_w, n_channels, n_classes):

    inputs = layers.Input(shape=(img_h, img_w, n_channels))
    s = Lambda(lambda x: x / 255)(inputs)     # Pas besoin si input déjà normalisé
    s = inputs
    
    model = EfficientNetB0(weights = None,   #tester avec imagenet (3 channels)
                           include_top = False,
                           input_shape = input_shape)(s)
    model.trainable = False
    
    x = layers.UpSampling2D(2)(model)   
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
             
    x = layers.UpSampling2D(2)(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(2)(x)

    # Add a per-pixel classification layer : output
    outputs = Conv2D(n_classes, 3, activation="softmax", padding="same")(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model
    
    
model = EfficientNet_model(img_h, img_w, n_channels, n_classes = 4)

model.summary()







































