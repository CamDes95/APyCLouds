import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import segmentation_models as sm
import os

#model = load_model('checkpoints/model_efficientnetb2_encoder_weights_imagenet_lr0.01_12epochs_DataAugmentEncoderFreeze_04_0.53.h5', custom_objects={'loss': bce_dice_loss}, compile=False)
model = load_model('checkpoints/model_densenet121_encoder_weights_imagenet_lr0.001_20epochs_DataAugmentEncoderFreeze_14_0.54.h5', custom_objects={'loss': bce_dice_loss}, compile=False)


#model = load_model('model_UNet_2.hdf5')

height=224
width=224
directory = "reduced_train_images_224/"
mask_path="reduced_train_masks_224/"
name_images = os.listdir(directory)
name_images=name_images[1:]

n_images = 800
ind_start = 4500
n_pixels = height * width
total_size = n_images * n_pixels

y_true = np.zeros((total_size,4))
y_pred = np.zeros((total_size,4))

for ind_from_0, index_image in enumerate(range(ind_start,ind_start + n_images)):
    print(ind_from_0)
    im = cloudImage.cloudImage(path=directory,
                               mask_path=mask_path,
                               fileName=name_images[index_image],
                               height=height, width=width)
    X = np.expand_dims(im.load(),axis=0)
    y = model.predict(X)

    masks=np.squeeze(im.load(is_mask=True))
    
    for k in range(4):
        y_true[n_pixels*ind_from_0:n_pixels*(ind_from_0+1),k] = (masks[:, :, k]).flatten()
        y_pred[n_pixels*ind_from_0:n_pixels*(ind_from_0+1),k] = np.squeeze(y[0, :, :, k]).flatten()            
        
plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')

classes = ['Fish', 'Flower', 'Gravel', 'Sugar']

for k in range(4):
    fpr, tpr, thresholds = roc_curve(y_true[:,k], y_pred[:,k])
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Classe {} (area = {:.3f})'.format(classes[k], auc_))
    plt.xlabel("False positive")
    plt.ylabel("True positive")
    plt.title("densenet121")

plt.legend()
plt.show()
