import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser as wb
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import bce_dice_loss
import cloudImage
import os
import cv2
from efficientnet.tfkeras import EfficientNetB4 # Pour charger le modèle
import time


# Chargement et mise en forme des données
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(csv, nrows):
    df = pd.read_csv(csv , nrows=nrows)
    return df;
df_train = load_data(csv = "train.csv", nrows=22184)

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()

sub_df = load_data(csv = 'sample_submission.csv', nrows =14792 )
sub_df['ImageID'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageID'].unique(), columns=['ImageID'])

# Barre latérale
st.markdown(
    f'''
    <style>
        .sidebar .sidebar-content {{
            width:700px;
        }}
    ''', unsafe_allow_html=True)  # Width de la sidebar
    
st.sidebar.header("Menu")
menu_list = ["Présentation du projet", "Dataset", "Modélisation", "Résultats", "Prédiction", "Application sur de nouvelles données"]
menu_sel = st.sidebar.radio("", menu_list, index=0, key=None)

######################### PRESENTATION PROJET ##########################

left, middle, right = st.beta_columns(3)
st.markdown('<style>.style1{color: black; font-size:50px; font-weight:bold}, </style>', unsafe_allow_html=True)
middle.markdown('<p class="style1"> APyClouds </p>',  unsafe_allow_html=True)

if menu_sel == "Présentation du projet":
    left, middle, right = st.beta_columns([1,12,1])
    middle.image("clouds_sky.jpg", width=600)
    
    st.subheader("Introduction")
    st.write("""Ce projet répond à une compétition mise en ligne sur le site Kaggle :
                Understanding Clouds from Satellites Images.""")
    st.write("""Les scientifiques de l'institut Max Plack de Météorologie développent et analysent 
             des modèles très complexes simulant les processus géophysiques ayant lieues dans 
             l'atmosphère. Les nuages superficiels jouent un rôle déterminant sur le climat de la 
             Terre, et sont difficiles à interpréter et représenter dans les modèles climatiques. Les chercheurs désirent améliorer 
             l’identification de ces types de nuages, ce qui permettrait de construire de meilleures projections et fournir des pistes de
             compréhension vis à vis du changement climatique global.""")
    st.write("""Au total, quatre formations nuageuses ont été identifiées par les scientifiques : Fish, Flower, Gravel et Sugar.  """)
    st.write("Afin de répondre à la problèmatique posée par les scientifiques de l'Institut Max Planck, nous avons construit un modèle de Deep Learning capable d'identifier les quatre formes de nuages.")

    st.subheader("Description des formations nuageuses :")
    
    if st.checkbox("Fish"):
        left, middle, right = st.beta_columns(3)
        left.image("Fish.png", width= 700)
        st.write("""Réseau de nuages sous forme de structures squelettiques allongées qui s'étendent parfois
jusqu'à 1000 km, pour la plupart longitudinalement.""")

    if st.checkbox("Flower"):
        left, middle, right = st.beta_columns(3)
        left.image("Flower.png", width = 700)
        st.write(""" Nuages stratiformes à grande échelle apparaissant sous forme de « bouquets », séparation
nette entre chaque « bouquet ».""")

    if st.checkbox("Gravel"):
        left, middle, right = st.beta_columns(3)
        left.image("Gravel.png", width = 700)
        st.write(""" Nuages de caractéristiques granulaires marqués par des arcs ou anneaux.
La différence entre les formations de type Gravel et Sugar est parfois difficile à identifier.""")
    
    if st.checkbox("Sugar"):
        left, middle, right = st.beta_columns(3)
        left.image("Sugar.png", width=700)
        st.write("""Zones étendues de nuages très fins ; en cas de fort débit, ils peuvent former de fines «
veines » ou « plumes » (nuages dentritiques).""")



######################### DATASET ##########################

from cloudImage2 import cloudImage2
from cloudImage import cloudImage
from catalogueImage import catalogueImage
path = "train_images/"

if menu_sel == "Dataset":
        
    # Exploration des images
    st.subheader("Visualisation des images du jeu de données")
    opt = st.selectbox(
        "Images",
        df_train.FileName.unique())
    path = "train_images/"
    st.image(path + opt)
 
    st.subheader('Les 10 premières lignes du dataset')
    st.dataframe(df_train.head(10))   
    
    st.subheader('Les colonnes du Dataset')
    st.write(df_train.columns)
    
    # Nombre de formes dans le dataset
    fish = df_train[df_train['PatternId'] == 'Fish'].EncodedPixels.count()
    flower = df_train[df_train['PatternId'] == 'Flower'].EncodedPixels.count()
    gravel = df_train[df_train['PatternId'] == 'Gravel'].EncodedPixels.count()
    sugar = df_train[df_train['PatternId'] == 'Sugar'].EncodedPixels.count()
    
    # plotting a pie chart
    
    fig, ax = plt.subplots(figsize=(6, 6))
    x = list([fish, flower, gravel, sugar])
    plt.pie(x, labels=['Fish', 'Flower', 'Gravel', 'Sugar'], autopct = '%1.1f%%')
    ax.set_title('Cloud Types');
    st.subheader("Répartition des formes nuageuses")
    st.pyplot(fig)
    
    df = pd.DataFrame([['Sugar',sugar],['Flower',flower],['Gravel',gravel],['Fish',fish]], columns= ['Pattern', 'Count'])

    if st.checkbox("Afficher le nombre d'images par forme nuageuse identifiée"):
        st.dataframe(df)

    if st.checkbox("Afficher les dimensions du dataset"):
        st.write(df_train.shape)
    
    st.markdown('<p class="style">Nous constatons que la forme Sugar est plus détectée que les autres, Flower étant celle qui est la moins observée par les scientifiques. </p>', unsafe_allow_html=True)
    
    st.write("Au total, 11836 formes de nuages ont été identifiées par les scientifiques sur les images d'entraînement.")    
        
    st.subheader('Visualiser les formes identifiés sur les images (Bounding Box)')
    #slider1 = st.slider("Choix du premier masque", 0, 100, 1)
    #slider2 = st.slider("Choix du deuxième masque", 0, 100, 1)
    #st.write("Masques :", slider1)
    #st.write("Masques :", slider2)
    viz = catalogueImage(dataFrame = df_train, indexes = range(1, 4))
    
    st.write(viz.visualizeCatalogue())
    
    
######################### MODELISATION ##########################  
  
if menu_sel == "Modélisation":
    st.markdown('<style>.stylex{color: black; font-size:35px}, </style>', unsafe_allow_html=True)
    st.subheader("Segmentation d'image")
    st.text(" ")
    st.markdown('<style> .text{color: black; font-size:22px}</style>', unsafe_allow_html=True)
    st.markdown("""Nous allons développer un modèle de Deep Learning en utilisant des réseaux de neurones convolutifs (CNN) et des méthodes de Transfer Learning. 
                """,unsafe_allow_html=True)
    
    st.markdown('Schéma type du processus à implémenter :', unsafe_allow_html=True)
    st.image('demarche.png', width=600)

    st.markdown("""Afin de classifier chacune des formes nuageuses, nous utiliserons les techniques de segmentation d'images à l'aide d'un Auto-Encoder (Encoder + Decoder).
                L'ensemble des démarches réalisées sont schématisées sur l'image ci-dessous.""",unsafe_allow_html=True)


    st.markdown('<style>.style2{color: black; font-size:25px}, </style>', unsafe_allow_html=True)
    st.text(" ")

    left, middle, right=st.beta_columns([8,1,1])
    left.image('cnn.png', width=600)

    if st.checkbox("Image inputs"):
        st.info("Nous travaillons sur un dataset de 5546 images de taille 1400x2100. Les images utilisées en entrée du modèle ont été formatées en 224x224.")

    if st.checkbox("Data Generator"):
        st.info("Un générateur de données permet de charger l’ensemble des données dans le réseau de neurones via des mini-lots d’images pour éviter de saturer la RAM ou Le GPU de l'ordinateur.")

    if st.checkbox("Image batchs"):
        st.info("Les lots d'images qui vont être traités par notre modèle (batch size = 16).")

    if st.checkbox("Data Augmentation"):
        st.info("Les techniques d’augmentation de données permettent d’accroître la diversité de notre jeu d’entraînement en appliquant des transformations aléatoires sur les images.")

    if st.checkbox("Encoder"):
        st.info("Un modèle CNN pré-entraîné utilisé par Transfer Learning (EfficientNet, ResNet, DenseNet) qui va extraire les composantes des images et compresser l'information.")

    if st.checkbox("Decoder"):
        st.info("un modèle CNN qui va convertir la sortie de l’encodeur et reconstruire l’image sous forme de classes.")

    if st.checkbox("Classification"):
        st.info("La couche de sortie correspond à une classification multi-classes par pixels. La fonction d'activation 'Softmax' permet de récupèrer les probabilités d'appartenance des pixels à chaque classe.")
    

######################### RESULTATS ##########################

if menu_sel == "Résultats":  
    st.subheader("Entraînement")
    st.markdown("""<p>Nous avons entraînés plusieurs modèles en faisant varier les hyper-paramètres afin de déterminer celui obtenant la meilleure performance. 
                Les résultats de ces entraînements sont détaillés dans le tableau ci-dessous. </p>""", unsafe_allow_html=True)
    st.markdown("""Liste des hyper-paramètres utilisés :""")
    
    if st.checkbox("Poids"):
        st.info("L’utilisation de modèles pré-entraînés comme encodeur permet de récupérer les poids et d’extraire précisément les caractéristiques des images. Dans notre projet, nous avons utilisés les poids de la base de données ImageNet.")

    if st.checkbox("Fonction de perte"):
        st.info("Nous allons ici utiliser la fonction de perte dice loss (indice de Sørensen-Dice) qui permet de mesurer la similarité entre deux échantillons.")

    if st.checkbox("Taux d'apprentissage"):
        st.info("Il permet de règler le rythme de mise à jour des poids à la fin de chaque batch. Nous testerons ici plusieurs taux d'apprentissage de 1e-2 à 5e-5 en utilisant l'optimiseur Adam.")

    if st.checkbox("Batch size"):
        st.info("En raison de la mémoire disponible sur notre matériel pour l'entraînement, nous avons utilisé un batch size de 16.")

    if st.checkbox("Fonction d'activation"):
        st.info("Nous avons utilisé la fonction d’activation softmax qui génère un vecteur représentant les distributions de probabilité d’une liste de résultats potentiels. Elle est souvent utilisée dans le cas des classifications multi-classes.")

    if st.checkbox("Nombre d'epochs"):
        st.info("Nous avons tout d'abord entraînés plusieurs modèles sur 10 epochs, puis sélectionnés ceux présentant les meilleurs performances afin de les réentraîner à l'aide d'un EarlyStopping.")

    st.markdown("Le tableau ci-dessous présente les résultats des deux meilleurs modèles entraînés.")
    
    st.image("final iteration.png")
    
    st.markdown("Au vu des performances, nous allons présenter dans la partie suivante les prédictions du modèle utilisant l'encodeur EfficientNetB2.")
    
    if st.checkbox("Liste et résultats des modèles testés"):
        st.image("iterations.png")
    

######################### PREDICTION ##########################

if menu_sel == "Prédiction":
    st.subheader("Prédiction   \n\n")
    
    model = load_model("model_efficientnetb2_encoder_weights_imagenet_lr0.001_20epochs_DataAugmentEncoderFreeze_07_0.57.h5", custom_objects={'loss': bce_dice_loss}, compile=False)

    images_list = ["Images d'entraînement", "Images test"]
    images_sel = st.radio("", images_list, index=0, key=None)
    
    patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']    
    # TRAIN IMAGES
    
    if images_sel == "Images d'entraînement":
        
        directory = "reduced_train_images_224/"
        left, right = st.beta_columns(2)
        opt = st.selectbox("Choix de l'image", df_train.FileName.unique())
        path = directory + opt
        left.image(path, channels="BGR", width = 341)
        
        
        with st.spinner("Image en cours de segmentation..."):
        
            im = cloudImage(path = "reduced_train_images_224/",
                                       mask_path="reduced_train_masks_224/",
                                       fileName = opt,
                                       height = 224,
                                       width = 224)
            X = np.expand_dims(im.load(),axis=0)
            y = model.predict(np.expand_dims(X, axis=3))                
            
           # for percent in range(100):
           #    time.sleep(0.03)
           #     bar.progress(percent+1)
           # st.success("Segmentation réussie!")
           
            pattern = st.selectbox("Choix de la forme à observer", patternList)

            threshold = st.slider('Choix du seuil de probabilité (optimal autour de 0.7)', min_value=0.0, max_value=1.0, value=0.7)
            
            fig = plt.figure(figsize=(30,30))
            ax = fig.subplots(1,1)
            if pattern == "Fish":
                ax.imshow(np.squeeze(y[0, :, :, 0]>threshold))
                ax.axis(False)
            elif pattern == "Flower":
                ax.imshow(np.squeeze(y[0, :, :, 1]>threshold))
                ax.axis(False)
            elif pattern == "Gravel":
                ax.imshow(np.squeeze(y[0, :, :, 2]>threshold))
                ax.axis(False)
            else:
                ax.imshow(np.squeeze(y[0, :, :, 3]>threshold))
                ax.axis(False)
            right.pyplot(fig, width = 100);
            
            st.success("Segmentation réussie !")
            
        if st.checkbox("Comparaison de la prédiction au masque d'entraînement"):
            masks=np.squeeze(im.load(is_mask=True))
            if pattern == "Fish":
                ax = plt.subplot(1,2,1)
                ax1 = plt.subplot(1,2,2)
                ax.imshow(np.squeeze(masks[:, :, 0]))
                ax1.imshow(np.squeeze(y[0, :, :, 0]>threshold))
                ax.axis(False)
                ax1.axis(False)
                ax.title.set_text("Fish - Mask") 
                ax1.title.set_text("Fish - Predicted") 
            elif pattern == "Flower":
                ax = plt.subplot(1,2,1)
                ax1 = plt.subplot(1,2,2)
                ax.imshow(np.squeeze(masks[:, :,1]))
                ax1.imshow(np.squeeze(y[0, :, :, 1]>threshold))
                ax.axis(False)
                ax1.axis(False)
                ax.title.set_text("Flower - Mask") 
                ax1.title.set_text("Flower - Predicted") 
            elif pattern == "Gravel":
                ax = plt.subplot(1,2,1)
                ax1 = plt.subplot(1,2,2)
                ax.imshow(np.squeeze(masks[:, :, 2]))
                ax1.imshow(np.squeeze(y[0, :, :, 2]>threshold))
                ax.axis(False)
                ax1.axis(False)
                ax.title.set_text("Gravel - Mask") 
                ax1.title.set_text("Gravel - Predicted") 
            else:
                ax = plt.subplot(1,2,1)
                ax1 = plt.subplot(1,2,2)
                ax.imshow(np.squeeze(masks[:, :, 3]))
                ax1.imshow(np.squeeze(y[0, :, :, 3]>threshold))
                ax.axis(False)
                ax1.axis(False)
                ax.title.set_text("Sugar - Mask") 
                ax1.title.set_text("Sugar - Predicted") 
            plt.rcParams.update({'font.size': 30})
            st.pyplot(fig, width = 250);
    
    # TEST IMAGES
    if images_sel == "Images test":
        
        left, right = st.beta_columns(2)
        directory1 = "reduced_test_images_224/"
        opt1 = st.selectbox(
        "Choix de l'image",
        test_imgs)
        path1 = directory1 + opt1
        left.image(path1, channels="BGR", width = 341)

        with st.spinner("Image en cours de segmentation..."):
            
            im = cloudImage(path = "reduced_test_images_224/",
                                       fileName = opt1,
                                       height = 224,
                                       width = 224)
            X = np.expand_dims(im.load(),axis=0)
            y = model.predict(np.expand_dims(X, axis=3))                        
            
            pattern = st.selectbox("Choix de la forme à observer", patternList)
            threshold = st.slider('Choix du seuil de probabilité (optimal autour de 0.7)', min_value=0.0, max_value=1.0, value=0.7)
            
            fig = plt.figure(figsize=(30,30))
            ax = fig.subplots(1,1)
            if pattern == "Fish":
                ax.imshow(np.squeeze(y[0, :, :, 0]>threshold))
                ax.axis(False)
            elif pattern == "Flower":
                ax.imshow(np.squeeze(y[0, :, :, 1]>threshold))
                ax.axis(False)
            elif pattern == "Gravel":
                ax.imshow(np.squeeze(y[0, :, :, 2]>threshold))
                ax.axis(False)
            else:
                ax.imshow(np.squeeze(y[0, :, :, 3]>threshold))
                ax.axis(False)
            right.pyplot(fig, width = 100);
            
            st.success("Segmentation réussie !")
 
    
######################### APPLICATION NEW DATA ########################## 
   
if menu_sel == "Application sur de nouvelles données":    
    st.subheader("Utilisation du modèle sur de nouvelles données")
    
    model = load_model("model_efficientnetb2_encoder_weights_imagenet_lr0.001_20epochs_DataAugmentEncoderFreeze_07_0.57.h5", custom_objects={'loss': bce_dice_loss}, compile=False)
    
    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img_resized = np.resize(np.asarray(opencv_image), (224,224))

        st.image(img_resized)

    patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
    pattern = st.selectbox("Choix de la forme à observer", patternList)
    
    with st.spinner("Image en cours de segmentation..."):
        
        X = img_resized
        y = model.predict(X)                

        threshold = st.slider('Choix du seuil de probabilité (optimal autour de 0.7)', min_value=0.0, max_value=1.0, value=0.7)
        
        fig = plt.figure(figsize=(30,30))
        ax = fig.subplots(1,1)
        if pattern == "Fish":
            ax.imshow(np.squeeze(y[0, :, :, 0]>threshold))
            ax.axis(False)
        elif pattern == "Flower":
            ax.imshow(np.squeeze(y[0, :, :, 1]>threshold))
            ax.axis(False)
        elif pattern == "Gravel":
            ax.imshow(np.squeeze(y[0, :, :, 2]>threshold))
            ax.axis(False)
        else:
            ax.imshow(np.squeeze(y[0, :, :, 3]>threshold))
            ax.axis(False)
        right.pyplot(fig, width = 100);
        
        st.success("Segmentation réussie !")
    
    
    
    
    
    
    
    
# Ajouter bouton pressoir repo git, Dataset et FAQ 

st.text("")
left_column, right_column = st.sidebar.beta_columns(2)
pressed = left_column.button('GitHub Repository')
pressed_2 = right_column.button("Kaggle  \n"
                                "Dataset")
if pressed:
    right_column.write(wb.open_new_tab("https://github.com/CamDes95/APyCLouds/tree/main"))

if pressed_2:
    left_column.write(wb.open_new_tab('https://www.kaggle.com/c/understanding_cloud_organization/data'))


linkedin_profile = ["https://fr.linkedin.com/in/camille-desjardin-82842b122",
                    "https://fr.linkedin.com/in/toufik-saddik-b2131640",
                    "https://fr.linkedin.com/in/luceglin" ]

st.sidebar.markdown("## About")
st.sidebar.info("Projet segmentation de régions nuageuses - Avril 2021  \n\n"
                "__________________  \n"
                "Luc EGLIN   \n"
                "{}   \n"
                "Camille DESJARDIN   \n"
                "{}   \n"
                "Toufik SADDIK   \n"
                "{}   \n".format(linkedin_profile[2],linkedin_profile[0],linkedin_profile[1]))



#FAQ
expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")    
    
