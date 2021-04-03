import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss
import time

model = load_model("model_EffNetB2_4_N_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)

# train images
directory = "reduced_train_images_224/"
name_images = os.listdir(directory)
name_images=name_images[1:]


# VIZ COMPARAISON AVEC TRAIN IMAGES

for index_image in range(20):
    im = cloudImage.cloudImage(path=directory,
                               mask_path="reduced_train_masks_224/",
                               fileName=name_images[index_image],
                               height=224, width=224)
    X = np.expand_dims(im.load(),axis=0)
    y = model.predict(np.expand_dims(X, axis=3))
    
    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(np.squeeze(im.load()))
    ax1.axis(False)
    plt.title("Original image")
    plt.suptitle("EfficientNetB0_1_N")
    
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
           
    plt.show()
    

# VIZ COMPARAISON AVEC TEST IMAGES

# test images
directory_test = "reduced_test_images_256/"
test_images = os.listdir(directory_test)
test_images=test_images[1:]

directory_test2 = "reduced_test_images_224/"
test_images2 = os.listdir(directory_test2)
test_images2 = test_images2[1:]

model = load_model("model_EffNetB2_4_N_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)
model2 = load_model("model_EffNetB0_4_N_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)


for index_image in range(20):
    # Pred modèle 1
    im1 = cloudImage.cloudImage(path=directory_test,
                               fileName=test_images[index_image],
                               height=256, width=256)
    X = np.expand_dims(im1.load(),axis=0)
    y = model.predict(np.expand_dims(X, axis=3))
    
    # Pred modèle 2
    im2 = cloudImage.cloudImage(path=directory_test2,
                               fileName=test_images2[index_image],
                               height=224, width=224)
    X2 = np.expand_dims(im2.load(),axis=0)
    y2 = model2.predict(np.expand_dims(X2, axis=3))
    
    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(np.squeeze(im1.load()))
    ax1.axis(False)
    plt.title("Original image")
    plt.suptitle("EfficientNetB2_4_N - B0_4_N \n dice_coef : 0.597 - 0.617")
    
    patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    for k in range(4):
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
            
            ax = plt.subplot(4, 3, 3*k + 3)
            ax.imshow(np.squeeze(y2[0, :, :, k]>.5))
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
           
    plt.show()