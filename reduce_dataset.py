# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021

@author: luc eglin, camille desjardin, toufik saddik

Creates directories with graylevel images and masks
in reduced size, used for segmentation models 
"""


import pandas as pd
import cloudImage
from time import time
import os

reduced_size = [256, 256]  # height x width

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)
t1 = time()

for index_image in range(n_images):
    t2 = time()
    if (index_image % 100) == 0:
        print(index_image, str(t2-t1))
        t1 = time()
    im = cloudImage.cloudImage(path="train_images/",
                               mask_path="reduced_train_masks_256/",
                               fileName=name_images[index_image],
                               dataFrame=df_train,
                               new_size=reduced_size)
    im.computeBoxCoordinates()
    im.saveReducedImageAsJPG("reduced_train_images_256/")
    im.saveMaskAsJPG()

validation_dir = "test_images/"
name_images = os.listdir(validation_dir)
n_images = len(name_images)
t1 = time()

for index_image in range(n_images):
    t2 = time()
    if (index_image % 100) == 0:
        print(index_image, str(t2-t1))
        t1 = time()
    im = cloudImage.cloudImage(path=validation_dir,
                               fileName=name_images[index_image],
                               new_size=reduced_size)

    im.saveReducedImageAsJPG("reduced_test_images_256/")

