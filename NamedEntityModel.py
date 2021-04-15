# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:22:26 2021

@author: luc_e
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, RNN, GRUCell, Embedding, Bidirectional, Dropout
import NamedEntityPreproc

NamedEntityPreproc
print(vocab_size)

#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

model = Sequential()
model.add(Embedding(vocab_size, vocab_size))
model.add(Bidirectional(RNN(GRUCell(256, recurrent_initializer='glorot_uniform'),
                                return_sequences=True)))
model.add(Dropout(0.3))
# la sortie a n_tags+1 units, c'est à dire le nombre de tags possibles plus la classe utililisée pour égalisée
# la taille des data (qui ne sera pas prise en compte dans l'entraînement grâce à la fonction de coût customisée)
model.add(Dense(n_tags+1, activation='softmax'))
model.summary()



loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def loss_function(real, pred):
    # Mask => vaudra 0 lorsque y vaut n_tags, c'est à dire la valeur avec laquelle on a compléter le tableau y_train
    mask = tf.math.logical_not(tf.math.equal(real, n_tags))
    # Pour respecter le type de y
    mask = tf.cast(mask, dtype=pred.dtype)
    # fonction de perte
    loss_ = loss_object(real, pred)
    # Apply mask on loss function
    loss_ *= mask
    # on renvoit la moyenne calculée en dehors de mask = 0, valeur que l'on ne veut pas prendre en compte ds le calcul
    return tf.reduce_mean(loss_)

model.compile(loss=loss_function, optimizer='adam')