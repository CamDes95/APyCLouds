import pandas as pd
import numpy as np
import cloudImage
import os
from time import time
import dataGeneratorFromClass
import model_UNet
import model_Unet2
import matplotlib.pyplot as plt

reduced_size = [144, 224]  # height x width

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)

for ntime in range(10):
    input("Press Enter to continue...")
    d = dataGeneratorFromClass.DataGenerator(list_IDs=np.arange(48), list_images=name_images, dim=reduced_size, batch_size=16)
    X, y = d.__getitem__(0)
    print(y.shape)
    print((np.squeeze(X)).shape, y.shape)
    plt.subplot(3,2,1)
    plt.imshow(np.squeeze(X[0,:,:]))
    plt.axis(False)
    for k in range(5):
        plt.subplot(3,2,k+2)
        plt.imshow(np.squeeze(y[0,:,:,k]))
        plt.axis(False)
        plt.title(str(np.unique(y[0,:,:,k])))

    plt.show()

print(y.max())