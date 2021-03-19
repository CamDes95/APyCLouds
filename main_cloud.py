#import os
#os.chdir("./Desktop/APyClouds/understanding_cloud_organization")

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
from modules import *

# Importation et modification dataset train 
df_train = pd.read_csv("train.csv")
   
df_train['ImageID']          =   df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId']         =   df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence']   = ~ df_train['EncodedPixels'].isna()

df_train.head()


# Importation submission et test img
sub_df = pd.read_csv('sample_submission.csv')
sub_df['ImageID'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageID'].unique(), columns=['ImageID'])
print(test_imgs)


# Première observation des masques de train
def decode(mask, shape=(1400, 2100)):
    m = mask.split()
    a = list()
    
    for x in (m[0:][::2], m[1:][::2]):
        a.append(np.asarray(x, dtype=int))
    starts, lengths = a
    starts -= 1
    stop = starts + lengths
    image = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for i, j in zip(starts, stop):
        image[i:j] = 1   
    image = image.reshape(shape, order='F') 
    return image

#Visualiser masques de train

train_img_path = "./train_images/"
plt.figure(figsize=[60, 30])
for i, row in df_train[:16].iterrows():
    img = cv2.imread(train_img_path +  row['ImageID'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc_pix = row['EncodedPixels']
    try:
        mask = decode(enc_pix)
    except:
        mask = np.zeros((1400, 2100))
    plt.subplot(4, 4, i+1)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6, cmap='gray')
    plt.title("Label %s" % row['PatternId'], fontsize=32)
    plt.axis('off')    
plt.show()



##### Séparation des variables

df_train2 = df_train.iloc[:3000, :]

mask_count_df = df_train.groupby('ImageID').agg(np.sum).reset_index()
mask_count_df.sort_values('PatternPresence', ascending=False, inplace=True)
print(mask_count_df.index)

# Séparation des variables train et val

train_idx, val_idx = train_test_split(mask_count_df.index, random_state=234, test_size=0.2) #ratify=strat


# Instanciation DataGenerator
train_generator = DataGenerator(
    train_idx, 
    df=mask_count_df,
    target_df=df_train,
    reshape=(320, 480),
    augment=True,
    n_channels=3,
    n_classes=4)

val_generator = DataGenerator(
    val_idx, 
    df=mask_count_df,
    target_df= df_train, 
    reshape=(320, 480),
    augment=False,
    n_channels=3,
    n_classes=4)

##### Définition du modèle UNet like #####
# TESTER AVEC UNET, RESNET

img_size=(320,480)
num_classes = 4

model = get_model(img_size, num_classes)
model.summary()

# Compilation  

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])
# model.compile(optimizer=Adam(lr = 3e-4), loss="binary_crossentropy", metrics=[dice_coef])
# Test avec binary_crossentropy, sparse_categorical_entropy


##### ENTRAINEMENT #####

history = model.fit_generator(
     train_generator,
     validation_data=val_generator,
     epochs=10,
     verbose=1)

# fonction de perte et dice_coeff
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(history.history["loss"], label = "dice loss")
plt.plot(history.history["val_loss"], label = "val dice loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
plt.legend()

plt.subplot(122)
plt.plot(history.history["dice_coef"], label="dice coef", color="red")
plt.plot(history.history["val_dice_coef"], label="val_dice coef", color="green")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("dice coef")
plt.show();

## Sauvegarde des poids du modèle et du modèle
model.save_weights("model2_weights.h5")
model.save("first_iteration_model2.h5", include_optimizer = False)




##### PREDICTION #####


### Stockage des pixels encodés dans sub_df 

best_threshold = 0.45
best_size = 15000

threshold = best_threshold
min_size = best_size

test_df = []
encoded_pixels = []
TEST_BATCH_SIZE = 500

for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
    batch_idx = list(range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE)))

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        gamma=0.8,
        base_path='./test_images/',
        target_df=sub_df,
        batch_size=1,
        n_classes=4)

    batch_pred_masks = model.predict_generator(test_generator,
                                               workers=1,
                                               verbose=1)

    # Predict out put shape is (320X480X4)
    # 4  = 4 classes, Fish, Flower, Gravel Surger.

    for j, idx in enumerate(batch_idx):
        filename = test_imgs['ImageID'].iloc[idx]
        image_df = sub_df[sub_df['ImageID'] == filename].copy()

        # Batch prediction result set
        pred_masks = batch_pred_masks[j, ]

        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[...,k].astype('float32')

            if pred_mask.shape != (350, 525):
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            pred_mask, num_predict = post_process(pred_mask, threshold, min_size)

            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)

sub_df['EncodedPixels'] = encoded_pixels

sub_df = sub_df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
sub_df.head()


## Export des résultats en csv
sub_df.to_csv("sample_submission_test2.csv")

# On garde slmt les lignes avec des masques
sub_df_2 = sub_df[sub_df['EncodedPixels'].notnull()]
sub_df_2["ImageID"] = sub_df["ImageID"]




"""
A TESTER 

## Tests visualisation
df_train["Label_EncodedPixels"] = df_train.apply(lambda row: (row["PatternId"], row["EncodedPixels"]), axis=1)
print(df_train)

# df avec label nuage et pixels associés
EncodedPixels_group = df_train.groupby("ImageID")["Label_EncodedPixels"].apply(list)
print(EncodedPixels_group)
"""


### Observation des résultats : masques de test
# Mauvaise visualisation, à résoudre !!!!!!

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
plt.figure(figsize=(50, 40))
for index, row in sub_df_2[:16].iterrows():
    img = cv2.imread("./test_images/%s" % row['ImageID'])[...,[2, 1, 0]]
    img = cv2.resize(img, (525, 350))
    mask_rle = row['EncodedPixels']
    mask = rle_decode(mask_rle)
    
    plt.subplot(4, 4, index+1)
    plt.imshow(img)
    plt.imshow(rle2mask(mask_rle, img.shape), alpha=0.5, cmap='gray')
    plt.title("Image %s" % (row['Image_Label']), fontsize=18)
    plt.axis('off')     
plt.show();

