# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Plot dice coeff, loss function from log and model
@author: luc eglin, camille desjardin, toufik saddik
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from bce_dice_loss import bce_dice_loss
import pandas as pd
import numpy as np

l_rates = ['1e-04','0.0005','0.001','0.005','0.01']
l_rates = ['0.001']

backbones = ['efficientnetb0','efficientnetb2','densenet121','resnet34']
backbones = ['efficientnetb2','densenet121']

n_epochs = 20
encoder_weights='imagenet'
df_res = pd.DataFrame({'Model':[],'learning rate':[],'loss':[],'vall_loss':[],'coef':[],'dice_coef':[]})

for index,BACKBONE in enumerate(backbones):
    for lr in l_rates:
        
        # Read log csv file
        df = pd.read_csv('log_'
                         +BACKBONE
                         +"_encoder_weights_"
                         +encoder_weights
                         +'_lr'+str(lr)
                         +'_'+str(n_epochs)
                         +'epochs_'
                         + 'DataAugment'+'.csv', 
                         sep=';')
        
        df_res.append({'Model':BACKBONE,
                      'learning rate':lr,
                      'min loss':min(df.loss),
                      'min val_loss':min(df.val_loss),
                      'coef':max(df.dice_coef),
                      'dice_coef':max(df.val_dice_coef)},
                      ignore_index=True)
        
        l = np.array(df.val_dice_coef)
        #plt.subplot(2,3,index+1)
        plt.plot(np.arange(len(df.loss)), l, label = 'lr='+lr);
        plt.legend();
        plt.xlabel("Epochs");
        plt.ylabel("val_dice_coef")
    plt.title("")
    plt.show()
    

for index,BACKBONE in enumerate(backbones):
    #for lr in l_rates:
        
    df = pd.read_csv('log_'
                     +BACKBONE
                     +"_encoder_weights_"
                     +encoder_weights
                     +'_lr'+str(lr)
                     +'_'+str(n_epochs)
                     +'epochs_'
                     + 'DataAugment'+'.csv', 
                     sep=';')
    
    df_res.append({'Model':BACKBONE,
                  'learning rate':lr,
                  'min loss':min(df.loss),
                  'min val_loss':min(df.val_loss),
                  'coef':max(df.dice_coef),
                  'dice_coef':max(df.val_dice_coef)},
                  ignore_index=True)
    
    l = np.array(df.loss)
    #plt.subplot(2,3,index+1)
    plt.plot(np.arange(len(df.loss)), l, label = BACKBONE);
    plt.legend();
    plt.xlabel("Epochs");
    plt.ylabel("Loss");
    #plt.title("")
plt.show()
    
for index,BACKBONE in enumerate(backbones):
    lr = '0.001'
        
    df = pd.read_csv('log_'
                     +BACKBONE
                     +"_encoder_weights_"
                     +encoder_weights
                     +'_lr'+str(lr)
                     +'_'+str(n_epochs)
                     +'epochs_'
                     + 'DataAugment'+'.csv', 
                     sep=';')

   # plt.subplot(2,3,index+1)
    plt.plot(np.arange(len(df.loss)), np.array(df.val_dice_coef), label = BACKBONE);
    #plt.plot(np.arange(len(df.loss)), np.array(df.dice_coef), label = 'Train/'+BACKBONE);
    plt.legend();
    plt.xlabel("Epochs");
    plt.ylabel("Val dice coeff");
#plt.title('Dice coeff')
plt.show()