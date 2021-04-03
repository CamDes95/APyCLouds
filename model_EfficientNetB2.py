import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D,Dropout,  MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
import efficientnet.tfkeras as efn 
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.efficientnet import preprocess_input

### PREPROCESSING

preprocess_input = preprocess_input

### IMAGE AUGMENTATION
img_augmentation = Sequential(
    [preprocessing.RandomRotation(factor=0.15),
     preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
     preprocessing.RandomFlip(),
     preprocessing.RandomContrast(factor=0.1),],)

################################################################
img_h = 256
img_w = 256
n_channels = 3
n_classes = 4



def EfficientNetB2_model(img_h, img_w, n_channels, n_classes = 4):
    
    encoder = EfficientNetB2(weights = None,
                             include_top = False,
                             input_shape = (img_h, img_w, 3))
    
    inputs = layers.Input(shape=(img_h, img_w, 3))
    #s = Lambda(lambda x: x / 255)(inputs)     # Pas besoin si input déjà normalisé
  # x = img_augmentation(inputs)
  # x = preprocess_input(x)

    model = encoder(inputs)
    model.trainable = False
    
    # Decoder
    x = layers.UpSampling2D(2)(model)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
             
    x = layers.UpSampling2D(2)(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(2)(x)

    # Add a per-pixel classification layer : output
    outputs = Conv2D(n_classes, (1,1), activation="softmax", padding="same")(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model
    
    
model = EfficientNetB2_model(img_h, img_w, n_channels, n_classes)

model.summary()



##########################################
"""
EfficientNetB0 : 224*224 

EfficientNetB1 : 240*240
EfficientNetB2 : 260*260
EfficientNetB3 : 300*300
EfficientNetB4 : 380*380
EfficientNetB5 : 456*456
EfficientNetB6 : 600*600

"""




































