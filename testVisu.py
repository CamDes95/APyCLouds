import pandas as pd
import numpy as np
import dataGeneratorFromClass
import matplotlib.pyplot as plt

reduced_size = [144, 224]  # height x width

df_train = pd.read_csv("train.csv")

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()
print(df_train.head())

name_images = df_train['FileName'].unique()
n_images = len(name_images)

for ntime in range(1):
    #input("Press Enter to continue...")
    d = dataGeneratorFromClass.DataGeneratorFromClass(list_IDs=np.arange(1,3), 
                                             list_images=name_images, 
                                             dim=reduced_size,
                                             batch_size=1,
                                             augment=True)
    X, y = d.__getitem__(0)
    print(y.shape)
    print((np.squeeze(X)).shape, y.shape)
    plt.subplot(3,2,1)
    plt.imshow(np.squeeze(X[0,:,:]))
    plt.axis(False)
    for k in range(4):
        plt.subplot(3,2,k+2)
        plt.imshow(np.squeeze(y[0,:,:,k]))
        plt.axis(False)
        plt.title(str(np.unique(y[0,:,:,k])))

    plt.show()

print(y.max())