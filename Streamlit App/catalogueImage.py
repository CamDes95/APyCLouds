import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudImage2 import cloudImage2
from cloudImage import cloudImage


df_train = pd.read_csv("train.csv")
df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()

class catalogueImage:
    def __init__(self, path = "train_images/", indexes = [], dataFrame=df_train):
        self.images_name = dataFrame['FileName'].unique()
        self.n_train_images = len(self.images_name)
        self.index_images_to_show = indexes
        self.n_images_to_show = len(self.index_images_to_show)
        self.im = []
        for _ in self.index_images_to_show:
            self.im.append(cloudImage2(path, self.images_name[_], dataFrame))
            
    def visualizeCatalogue(self):
        col_max = 3
        n_row = (self.n_images_to_show-1)//col_max + 1
        n_col = self.n_images_to_show if self.n_images_to_show < col_max else col_max

        # fig = plt.figure(figsize=(12,3*n_row))
        fig = plt.figure(figsize=(30,8*n_row)) # widht, height
        
        for ind_im, im in enumerate(self.im):
            plt.subplot(n_row, n_col, ind_im + 1)
            # Visualisation image
            im.visualize()
            # building mask
            im.computeBoxCoordinates()
            # Visualize mask
            im.visualizeBoxes()