import numpy as np
import tensorflow.keras
import cloudImage
import tensorflow as tf
import pandas as pd

df_train = pd.read_csv("train.csv")
"""
df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()

df_train.iloc[:, 2:].values
df_train["PatternId"]= df_train["PatternId"].replace(["Fish", "Flower", "Gravel", "Sugar"], [1,2,3,4])
df_train["PatternPresence"] = df_train["PatternPresence"].replace({"False":0,"True":1})
img_2_ohe_vector = {img:vec for img, pattern, vec in zip(df_train['FileName'], df_train["PatternId"], df_train.iloc[:,4].values)}
"""

####### TEST PR ROC #######
df_train = df_train[~df_train['EncodedPixels'].isnull()]
df_train['Image'] = df_train['Image_Label'].map(lambda x: x.split('_')[0])
df_train['Class'] = df_train['Image_Label'].map(lambda x: x.split('_')[1])
classes = df_train['Class'].unique()
df_train = df_train.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    df_train[class_name] = df_train['Class'].map(lambda x: 1 if class_name in x else 0)
df_train.head()

# dictionary for fast access to ohe vectors
img_2_ohe_vector = {img:vec for img, vec in zip(df_train['Image'], df_train.iloc[:, 2:].values)}





class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 batch_size=32,
                 dim=(1400,2100),
                 n_classes=4,
                 n_channels = 3,
                 shuffle=True,
                 dir_image="reduced_train_images_224/",
                 dir_mask="reduced_train_masks_224/",
                 list_images = []): 

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
        self.n_channels = n_channels
        self.len = int(np.floor(len(self.list_IDs) / self.batch_size))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len #int(np.floor(len(self.list_IDs) / self.batch_size))

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

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3), dtype=int)
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            im = cloudImage.cloudImage( mask_path = self.dir_mask,
                                        path=self.dir_image,
                                        fileName=self.list_images[ID],
                                        height=self.dim[0],
                                        width=self.dim[1])
            for k in range(3):
                X[i,:,:,k] = im.load() # chargement img réduite

            # Store mask
            y[i,:,:,:] = im.load(is_mask=True) # chgt mask réduit

        return X,y
    
    
    def get_labels(self):
        if self.shuffle:
            images_current = self.list_images[:self.len*self.batch_size]
           # images_current = self.list_images[:(int(np.floor(len(self.list_IDs) / self.batch_size)))*self.batch_size]
           #images_current = images_current[4501:5501]
            labels = [img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels).astype("float")
    
    