# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Datagenerator, based on the class cloudImage
@author: luc eglin, camille desjardin, toufik saddik
"""

import numpy as np
import tensorflow.keras
import cloudImage
import matplotlib.pyplot as plt
from skimage import transform


class DataGeneratorFromClass(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 batch_size=32,
                 dim=(1400,2100),
                 n_classes=4,
                 shuffle=True,
                 dir_image="reduced_train_images/",
                 dir_mask="reduced_train_masks/",
                 list_images = [],
                 augment=False):

        'Initialization'
        self.dim = dim
        self.dir_image = dir_image
        self.dir_mask = dir_mask
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.list_images = list_images
        self.augment = augment

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __random_transform(self, img, mask):
        boolean_flip = np.random.choice(a=[False, True], size=(2))
        random_angle = np.random.uniform(-20,20)
        
        if boolean_flip[0]:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        if boolean_flip[1]:
            img = np.flipud(img)
            mask = np.flipud(mask)    
            
        aug_img = transform.rotate(img, random_angle, mode='symmetric')
        aug_img = ((aug_img - aug_img.min()) * (1/(aug_img.max() - aug_img.min()) * 255)).astype('uint8')
        aug_mask = transform.rotate(mask, random_angle, order = 0, mode='symmetric')
        aug_mask = ((aug_mask - aug_mask.min()) * (1/(aug_mask.max() - aug_mask.min()))).astype('uint8')
        
        return aug_img, aug_mask
    
    def __augment_batch(self, img_batch, masks_batch):
        # loop on the batch
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(img_batch[i, ], masks_batch[i, ])  
            
        return img_batch, masks_batch

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=int)
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            im = cloudImage.cloudImage( mask_path = self.dir_mask,
                                        path=self.dir_image,
                                        fileName=self.list_images[ID],
                                        height=self.dim[0],
                                        width=self.dim[1])

            # load image
            X[i,] = im.load()

            # Store mask
            y[i,] = im.load(is_mask=True)

        if self.augment:
            X, y = self.__augment_batch(X, y)

        y = np.float32(y)
        X = np.float32(X)

        return X,y