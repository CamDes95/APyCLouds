import numpy as np
import keras
import cloudImage

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 batch_size=32,
                 dim=(1400,2100),
                 n_classes=4,
                 shuffle=True,
                 dir_image="reduced_train_images/",
                 dir_mask="reduced_train_masks/",
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

            X[i,:,:] = im.load()

            # Store mask
            y[i,:,:,:] = im.load(is_mask=True)

        return X,y