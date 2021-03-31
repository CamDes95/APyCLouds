
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
    [preprocessing.RandomRotation(factor=0.15),
     preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
     preprocessing.RandomFlip(),
     preprocessing.RandomContrast(factor=0.1),],
    name="img_augmentation",)

################################################################
img_h = 224
img_w = 224
n_channels = 3
n_classes = 4

def EfficientNet_model(img_h, img_w, n_channels, n_classes = 4):

    inputs = layers.Input(shape=(img_h, img_w, 3))
    #s = Lambda(lambda x: x / 255)(inputs)     # Pas besoin si input déjà normalisé
    
    x = img_augmentation(inputs)
    
    model = EfficientNetB0(weights = "imagenet",   #tester avec imagenet (3 channels)
                           include_top = False,
                           input_shape = (img_h, img_w, 3))(x)
    model.trainable = False
    
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
    #outputs = Conv2D(n_classes, (1,1), activation="softmax", padding="same")(x)
    outputs = Conv2D(n_classes, (3,3), activation="sigmoid", padding="same")(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model
    
    
model = EfficientNet_model(img_h, img_w, n_channels, n_classes)

model.summary()



##########################################
"""
EfficientNetB0 : 224*224 
LR : 1e-3 ok
LR : 1e-4 moins performant
LR : 1e-5 à tester


EfficientNetB1 : 240*240
EfficientNetB2 : 260*260
EfficientNetB3 : 300*300
EfficientNetB4 : 380*380
EfficientNetB5 : 456*456
EfficientNetB6 : 600*600

"""



































