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
from efficientnet.tfkeras import EfficientNetB4


# Chargement et mise en forme des données
#@st.cache
def load_data(nrows):
    df = pd.read_csv("train.csv", nrows=nrows)
    return df
df_train = load_data(nrows=22184)

df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()

# Barre latérale
st.sidebar.header("Menu")

######################### PRESENTATION PROJET ##########################

left, middle, right = st.beta_columns(3)
middle.title("APyClouds")

if st.sidebar.checkbox("Présentation du projet"):
    st.image("clouds_sky.jpg", width=600)
    
    st.title("Introduction")
    st.write("""Ce projet répond à une compétition mise en ligne sur le site Kaggle :
                Understanding Clouds from Satellites Images.""")
    st.write("""Les scientifiques de l'institut Max Plack de Météorologie développent et analysent 
             des modèles très complexes simulant les processus géophysiques ayant lieues dans 
             l'atmosphère. Les nuages superficiels jouent un rôle déterminant sur le climat de la 
             Terre, et sont difficiles à interpréter et représenter dans les modèles climatiques.""")
    st.write(""" Les chercheurs désirent améliorer l’identification de ces types de nuages, ce qui 
             permettrait de construire de meilleures projections et fournir des pistes de
             compréhension vis à vis du changement climatique global.""")
    st.write("""Au total, quatre formations nuageuses ont été identifiées par les scientifiques : Fish, Flower, Gravel et Sugar.  """)
    
    st.image("Fish.png")
    st.write("""Fish : Réseau de nuages sous forme de structures squelettiques allongées qui s'étendent parfois
jusqu'à 1000 km, pour la plupart longitudinalement.""")

    st.image("Flower.png")
    st.write("""Flower : Nuages stratiformes à grande échelle apparaissant sous forme de « bouquets », séparation
nette entre chaque « bouquet ».""")
    
    st.image("Gravel.png")
    st.write("""Gravel : Nuages de caractéristiques granulaires marqués par des arcs ou anneaux.
La différence entre les formations de type Gravel et Sugar est parfois difficile à identifier.""")

    st.image("Sugar.png")
    st.write("""Sugar : Zones étendues de nuages très fins ; en cas de fort débit, ils peuvent former de fines «
veines » ou « plumes » (nuages dentritiques).""")


    st.write("Afin de répondre à la problèmatique, nous avons crées un modèle de Deep Learning capable d'identifier les quatres formes de nuages.")

    st.markdown("Problématique")


######################### DATASET ##########################

from cloudImage2 import cloudImage2
from catalogueImage import catalogueImage
path = "train_images/"

if st.sidebar.checkbox("Dataset"):
    
    if st.checkbox("Afficher le jeu de données"):
        st.write(df_train.head(5))
        
    if st.checkbox("Informations sur le dataframe"):
        buffer = io.StringIO()
        df_train.info(buf=buffer)
        s = buffer.getvalue()
        with open(df_train, encoding="utf-8") as f:
            f.write(s)
    
    # Exploration des images
    st.subheader("Visualisation des images du jeu de données")
    opt = st.selectbox(
        "Images",
        df_train.FileName.unique())
    path = "train_images/"
    st.image(path + opt)

    
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    
    st.subheader('Les 10 premiers lignes du dataset')
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
    plt.pie(x, labels=['Fish', 'Flower', 'Gravel', 'Sugar'], autopct='%1.1f%%')
    ax.set_title('Cloud Types');
    st.subheader("Répartition des formes nuageuses dans notre dataset")
    st.pyplot(fig)
    st.markdown('<p class="style">Nous constatons que la forme Sugar est plus détectée que les autres, Flower étant celle qui est la moins observée par les scientifques </p>', unsafe_allow_html=True)
    
    df = pd.DataFrame([['Sugar',sugar],['Flower',flower],['Gravel',gravel],['Fish',fish]], columns= ['Pattern', 'Count'])

    st.write("Au total, 11836 fomres de nuages ont été identifiées par les scientifiques sur les images d'entraînement")
    if st.checkbox("Afficher le nombre d'images par formes nuageuses identifiées"):
        st.dataframe(df)

    if st.checkbox("Afficher les dimensions du dataset"):
        st.write(df_train.shape)
        
    st.subheader('Visualiser les formes identifiés sur les images (Bounding Box)')
    #slider1 = st.slider("Choix du premier masque", 0, 100, 1)
    #slider2 = st.slider("Choix du deuxième masque", 0, 100, 1)
    #st.write("Masques :", slider1)
    #st.write("Masques :", slider2)
    viz = catalogueImage(dataFrame = df_train, indexes = range(1, 10))
    
    st.write(viz.visualizeCatalogue())
    
    
######################### MODELISATION ##########################  
  
if st.sidebar.checkbox("Modélisation"):
    st.title("Modélisation") 
    st.markdown('<style>.style1{color: blue; font-size:35px}, </style>', unsafe_allow_html=True)

    st.markdown('<p class="style1">Modelisation Encoder-Decoder </p>', unsafe_allow_html=True)



    st.markdown('<style>.style2{color: black; font-size:25px}, </style>', unsafe_allow_html=True)

    st.markdown('<p class="style2">Schéma type du processus à implémenter. </p>', unsafe_allow_html=True)

    st.image('demarche.png', width=600)



    st.markdown('<style> .text{color: black; font-size:22px}</style>', unsafe_allow_html=True)

    st.markdown('<p class="text">Nous allons développer un modèle de Deep Learning en utilisant le Transfer Learning et un réseau de neurone convolutif (CNN),  qui va détecter les quatre types de formations nuageuses.</p>',unsafe_allow_html=True)

    st.text("")

    st.image('cnn.png', width=600)



    if st.checkbox("Image inputs"):

        st.write("Nous avons 5546 images de taille : (1400*2100). Les images utilisées en entrée du modèle ont été formatées en 224*224")

    if st.checkbox("Dta Generator"):

        st.write("Un générateur de données permet de charger l’ensemble des données dans le réseau de neurones via des mini-lots d’images pour éviter de saturer la RAM ou Le GPU de l'ordinateur.")

    if st.checkbox("Image batchs"):

        st.write("Les lots d'images qui vont être traités par notre modèle (batch size = 16.")

    if st.checkbox("Data Augmentation"):

        st.write("Les techniques d’augmentation de données permettent d’accroître la diversité de notre jeu d’entraînement en appliquant des transformations aléatoires sur les images.")

    if st.checkbox("Encoder"):

        st.write("Un modèle CNN pré-entraîné (UNet, EfficientNet, ResNet, DenseNet) qui va compresser l'information.")

    if st.checkbox("Decoder"):

        st.write("un modèle CNN qui va convertir la sortie de l’encodeur et reconstruire l’image sous forme de classes.")

    if st.checkbox("Classification"):

        st.write("La couche de sortie correspond à une classification multi-classes par pixels. La fonction d'activation 'Softmax' permet de récupèrer les probabilités d'appartenance des pixels à chaque classe.")

    st.markdown("<h2>Entraînement</h2>", unsafe_allow_html=True)

    st.markdown("<p>Nous avons entraînés plusieurs modèles afin de déterminer le plus stable et possèdant la meilleure performance. Les résultats de ces entraînements sont indiqués dans la partie suivante </p>", unsafe_allow_html=True)
    

######################### RESULTATS ##########################

if st.sidebar.checkbox("Résultats"):  
    st.title("Résultats")
    st.markdown("Présentation des résultats") 
    

######################### PREDICTION ##########################

if st.sidebar.checkbox("Prédiction"):
    st.title("Prédiction")
    
    model = load_model("model_efficientnetb2_encoder_weights_imagenet_lr0.001_20epochs_DataAugmentEncoderFreeze_07_0.57.h5", custom_objects={'loss': bce_dice_loss}, compile=False)

    directory = "reduced_train_images_224/"
    
    left, middle, right = st.beta_columns(3)
    
    opt = st.selectbox(
        "Image à prédire",
        df_train.FileName.unique())
    path = directory + opt
    middle.image(path, channels="BGR")
    
    with st.spinner("Image en cours de segmentation..."):
        im = cloudImage.cloudImage(path = "reduced_train_images_224/",
                                   mask_path="reduced_train_masks_224/",
                                   fileName = opt,
                                   height = 224,
                                   width = 224)
        X = np.expand_dims(im.load(),axis=0)
        y = model.predict(np.expand_dims(X, axis=3))
        
        masks=np.squeeze(im.load(is_mask=True))
        
        patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
        pattern = st.selectbox(
            "Pattern",
            patternList)
        
        threshold = st.slider('Choix du seuil de probabilité', min_value=0.0, max_value=1.0, value=0.05)
        
        fig = plt.figure(figsize=(8,8))
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
        middle.pyplot(fig, width = 250);

    
######################### APPLICATION NEW DATA ########################## 
   
if st.sidebar.checkbox("Application sur de nouvelles données"):    
    st.subheader("Utilisation du modèle sur de nouvelles données")
    
    uploaded_file = st.file_uploader("train_images/", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        
        ## Convertir image en 224*224
    
    
    
    
    
    
    
    
    
# Ajouter bouton pressoir repo git et FAQ 
left_column, middle_column, right_column = st.sidebar.beta_columns(3)
pressed = middle_column.button('GitHub Repository')
if pressed:
    right_column.write(wb.open_new_tab("https://github.com/CamDes95/APyCLouds/tree/main"))

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    