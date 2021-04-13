import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cloudImage
from bce_dice_loss import bce_dice_loss
import time
import cv2
from PIL import Image
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve
import plotly.express as px


model = load_model("model_EffNetB2_4_N2_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)

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
    plt.suptitle("EfficientNetB2_1 - dice_coef = 0.569 - loss = 0.74")
    
    patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
    masks=np.squeeze(im.load(is_mask=True))
    print(masks.shape, y.shape)
    
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
directory_test = "reduced_test_images_224/"
test_images = os.listdir(directory_test)
test_images=test_images[1:]

directory_test2 = "reduced_test_images_256/"
test_images2 = os.listdir(directory_test2)
test_images2 = test_images2[1:]

model = load_model("model_EffNetB0_4_TOP_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)
model2 = load_model("model_EffNetB2_4_N2_imgnet.h5", custom_objects={'loss': bce_dice_loss}, compile=False)


for index_image in range(20):
    # Pred modèle 1
    im1 = cloudImage.cloudImage(path=directory_test,
                               fileName=test_images[index_image],
                               height=224, width=224)
    X = np.expand_dims(im1.load(),axis=0)
    y = model.predict(np.expand_dims(X, axis=3))
    
    # Pred modèle 2
    im2 = cloudImage.cloudImage(path=directory_test2,
                               fileName=test_images2[index_image],
                               height=256, width=256)
    X2 = np.expand_dims(im2.load(),axis=0)
    y2 = model2.predict(np.expand_dims(X2, axis=3))
    
    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(np.squeeze(im1.load()))
    ax1.axis(False)
    plt.title("Original image")
    plt.suptitle("model_EffNetB0_4_TOP_imgnet - B2_4_N2 \n dice_coef : 0.6067 - 0.6081")
    
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
           
    plt.show();
    
    
    
    
    
    
    
    
############# AUC & ROC Curve ###############
# OU PR CURVE CAR DESEQUILIBRE DES CLASSES

model = load_model("model_densenet121_encoder_weights_imagenet_lr0.001_100epochs_DataAugmentEncoderFreeze.hdf5", custom_objects={'loss': bce_dice_loss}, compile=False)

### TEST SET ###

# prédiction = proba de y_test (val_gen)
y_pred = model.predict(val_gen)
y_pred2 = y_pred[:,0,0,:].reshape(992,4)

# Si la proba est > à 0.2 classé 1, sinon 0
pred = np.where(y_pred2 > 0.252,0,1).astype("float")

# récupération labels y_test 
y_true = val_gen.get_labels()
n_classes = y_true.shape[1]


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area (class imbalanced)
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure(figsize=(12,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-avg ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class - test set')
plt.legend(loc="lower right")
plt.show()

### Modification du seuil AUC

m1 = AUC(thresholds=[0.0, 0.5, 1.0])
m1.update_state(y_true, y_pred2)

# when threshold=0.0 , y_class always be 1
# when threshold=0.5 , y_class always be 0
# when threshold=1.0 , y_class always be 0 

print('AUC={}'.format(m1.result().numpy())) 
# output: AUC=0.5


#--------------------------------------------------------------
m2 = AUC(thresholds=[0.0, 0.0045, 1.0])
m2.update_state(y_true, y_pred2)

# when threshold=0.0    , y_class always be 1 
# when threshold=0.0045 , y_class will   be [0, 0, 0, 0, 1, 1]
# when threshold=1.0    , y_class always be 0 

print('AUC={}'.format(m2.result().numpy())) 
# output: AUC=0.75


#--------------------------------------------------------------
m3 = AUC(num_thresholds=200) 
# print(m3.thresholds)
m3.update_state(y_true, y_pred2)

print('AUC={}'.format(m3.result().numpy())) 
# output: AUC=0.875



### TRAIN SET ###

y_train = model.predict_generator(data_gen)
y_train2 = y_train[:,0,0,:].reshape(4496,4)

# récupération labels y_test 
y_true_train = data_gen.get_labels()
n_classes = y_true_train.shape[1]


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_train[:, i], y_train2[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_train.ravel(), y_train2.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve on train set

plt.figure(figsize=(12,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class - train set')
plt.legend(loc="lower right")
plt.show()



    
    
########   PR CURVE   ######### CAR DESEQ DES CLASSES

# Si la proba est > à 0.2 classé 1, sinon 0
pred = np.where(y_pred2 > 0.252,0,1).astype("float")

# récupération labels y_test 
y_true = val_gen.get_labels()
n_classes = y_true.shape[1]


y_score = model.predict(val_gen)
y_pred2 = y_score[:,0,0,:].reshape(992,4)

precision, recall, thresholds = precision_recall_curve(y, y_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()
    
    
    
    
    
    
    
    
    