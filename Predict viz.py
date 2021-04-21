# Enleve tous les messages de debuggage de tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss

model = load_model('model_ResNet50V2_3_30_4500.h5', custom_objects={'loss': bce_dice_loss}, compile=False)

# train images
directory = "reduced_train_images/"
name_images = os.listdir(directory)
name_images=name_images[1:]


# VIZ COMPARAISON AVEC TRAIN IMAGES

for index_image in range(20):
    im = cloudImage.cloudImage(path=directory,
                               mask_path="reduced_train_masks/",
                               fileName=name_images[index_image],
                               height=224, width=224)
    img = im.load() # shape (224, 224)
    X = np.expand_dims(img,axis=0) # shape (1, 224, 224)
    X = np.stack((X,)*3, axis=3) # shape (1, 224, 224, 3)
    print(X.shape)
    y = model.predict(X)
    
    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(np.squeeze(im.load()))
    ax1.axis(False)
    plt.title("Original image")
    plt.suptitle("ResNet50V2_1_N")
    
    patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
    masks=np.squeeze(im.load(is_mask=True))
    print(masks.shape, y.shape)
    
    for k in range(4):
        for i in patternList:
            ax = plt.subplot(4, 3, 3*k + 2)
            ax.imshow(np.squeeze(y[0, :, :, k]>.5))
            ax.axis(False)
            if k == 0:
                ax.title.set_text("Fish - Predicted")
            if k == 1:
                ax.title.set_text("Flower - Predicted")
            if k == 2:
                ax.title.set_text("Gravel - Predicted")
            if k == 3:
                ax.title.set_text("Sugar - Predicted") 
            #plt.title("Predicted class")
                
            ax2 = plt.subplot(4, 3, 3*k + 3)
            ax2.imshow(np.squeeze(masks[:, :, k]))
            ax2.axis(False)
            if k == 0:
                ax2.title.set_text("Fish - True")
            if k == 1:
                ax2.title.set_text("Flower - True")
            if k == 2:
                ax2.title.set_text("Gravel - True")
            if k == 3:
                ax2.title.set_text("Sugar - True") 
            #ax2.set_title("True class")
    plt.savefig('ResNet50V2_valid_5.png')
           
    plt.show()
    