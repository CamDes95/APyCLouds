# https://youtu.be/XyX5HNuv-xE
# https://youtu.be/q-p8v1Bxvac
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda, Activation
from tensorflow.keras.layers.experimental import preprocessing 
from tensorflow.keras.models import Sequential

# On modifie la précision flottante de 32-bits à 16-bits
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=1400, IMG_WIDTH=2100, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #s = tf.image.random_flip_left_right(s)
    #s = tf.image.random_flip_up_down(s)
    #s = tf.keras.preprocessing.image.random_rotation(s, 30)
    #s = tf.keras.preprocessing.image.random_shift(wrg=0.1, hrg=0.1)

    # Contraction path
    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1) 
    
    c1 = Dropout(0.1)(c1)

    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1) 

    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2) 

    c2 = Dropout(0.1)(c2)

    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2) 

    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3) 

    c3 = Dropout(0.2)(c3)

    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3) 

    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4) 

    c4 = Dropout(0.2)(c4)

    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4) 

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5) 

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5) 

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    print(u6.shape)
    print(c1.shape, c2.shape, c3.shape, c4.shape, c5.shape)
    print(p1.shape, p2.shape, p3.shape, p4.shape)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6) 

    c6 = Dropout(0.2)(c6)

    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6) 

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7) 

    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7) 

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8) 

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8) 

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9) 

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9) 

    outputs = Conv2D(n_classes, (1, 1))(c9)
    # Il faut rajouter une couche pour pouvoir utiliser la precision mixte
    # L'activation softmax necessite une precision de 32 bits
    outputs = Activation('softmax', dtype='float32', name='predictions')(outputs) 

    model = Model(inputs=[inputs], outputs=[outputs])

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model