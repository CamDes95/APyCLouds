# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Visualize prediction/true masks, from recorded model
@author: luc eglin, camille desjardin, toufik saddik
"""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss
import time
import segmentation_models as sm


model = load_model('checkpoints/model_efficientnetb2_encoder_weights_imagenet_lr0.01_12epochs_DataAugmentEncoderFreeze_04_0.53.h5', custom_objects={'loss': bce_dice_loss}, compile=False)


height=224
width=224
directory = "reduced_train_images_224/"
mask_path="reduced_train_masks_224/"
name_images = os.listdir(directory)
name_images=name_images[1:]


for index_image in range(1560,1590):
    im = cloudImage.cloudImage(path=directory,
                               mask_path=mask_path,
                               fileName=name_images[index_image],
                               height=height, width=width)
    X = np.expand_dims(im.load(),axis=0)
    y = model.predict(X)

    plt.subplot(4, 3, 1)
    plt.imshow(np.squeeze(im.load()))
    plt.axis(False)

    masks=np.squeeze(im.load(is_mask=True))
    print(masks.shape, y.shape)
    for k in range(4):
        plt.subplot(4, 3, 3*k + 2)
        plt.imshow(np.squeeze(y[0, :, :, k]>.5))
        plt.axis(False)
        #if k==0: plt.title("Predicted class")
        plt.subplot(4, 3, 3*k + 3)
        plt.imshow(np.squeeze(masks[:, :, k]))
        plt.axis(False)
        #if k == 0: plt.title("True class")
  
    
    plt.show()


    print("something")
    time.sleep(2.)
