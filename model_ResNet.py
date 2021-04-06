from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers

################################################################
img_h = 224
img_w = 224
n_channels = 3
n_classes = 4

def ResNet_model(img_h, img_w, n_channels, n_classes):

    inputs = layers.Input(shape=(img_h, img_w, n_channels))
   
    encoder = ResNet50V2(weights = "imagenet",   
                           include_top = False,
                           input_shape = (img_h, img_w, n_channels))(inputs)
                               
    x = layers.UpSampling2D(2)(encoder)   
    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
             
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(2)(x)

    outputs = layers.Conv2D(n_classes, n_channels, activation="softmax", padding="same")(x)
    
    model = Model(inputs = inputs, outputs = outputs)

    # Freeze des layers du ResNet transférés
    model.layers[1].trainable = False
    return model
    
    
model = ResNet_model(img_h, img_w, n_channels, n_classes)

model.summary()

##########################################
