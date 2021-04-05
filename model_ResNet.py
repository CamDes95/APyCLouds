import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D,Dropout,  MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
import efficientnet.tfkeras as efn 
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras import Sequential

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


### IMAGE AUGMENTATION A TESTER


################################################################
img_h = 224
img_w = 224
n_channels = 3
n_classes = 4

def ResNet_model(img_h, img_w, n_channels, n_classes = 4):

    inputs = layers.Input(shape=(img_h, img_w, 3))
   
    model = ResNet101V2(weights = "imagenet",   
                           include_top = False,
                           input_shape = (img_h, img_w, 3))(inputs)
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
    outputs = Conv2D(n_classes, (1,1), activation="softmax", padding="same")(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model
    
    
model = ResNet_model(img_h, img_w, n_channels, n_classes)

model.summary()



##########################################
"""



"""



































