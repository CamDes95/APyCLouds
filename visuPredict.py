import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss
import time

model = load_model("model_UNet_aug_size.hdf5", custom_objects={'loss': bce_dice_loss}, compile=False)
#model = load_model('model_UNet_2.hdf5')

directory = "reduced_train_images_3/"
name_images = os.listdir(directory)
name_images=name_images[1:]


for index_image in range(20):
    im = cloudImage.cloudImage(path=directory,
                               mask_path="reduced_train_masks_3/",
                               fileName=name_images[index_image],
                               height=352, width=528)
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
        if k==0: plt.title("Predicted class")
        plt.subplot(4, 3, 3*k + 3)
        plt.imshow(np.squeeze(masks[:, :, k]))
        plt.axis(False)
        if k == 0: plt.title("True class")
  
    
    plt.show()


    print("something")
    time.sleep(2.)