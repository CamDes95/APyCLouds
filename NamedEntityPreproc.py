# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:18:59 2021

@author: luc_e
"""

print("Test")
import pandas as pd
df = pd.read_csv("ner_dataset.csv", encoding="latin1")
df = df.fillna(method="ffill")
df.head(7)


# Replace tag by integers
modes_tag = df.Tag.unique()
n_tags = len(modes_tag)
tag2idx={t:i for i,t in enumerate(modes_tag)}
df['Tag'] = df.Tag.replace(tag2idx)
df.head(7)


print(df.shape)
df = df.drop(['POS'], axis=1)
df = df.groupby('Sentence #').agg(list)
df = df.reset_index(drop=True)
df.head(7)

len_max = max(df.Word.apply(len))
print("Maximum size of a sentence :", len_max)

from sklearn.model_selection import train_test_split
X_text_train, X_text_test, y_train, y_test = train_test_split(df.Word, df.Tag, test_size=0.2, random_state=1234)

y_test

import tensorflow as tf
# Définition du tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
# Mettre à jour le dictionnaire du tokenizer
tokenizer.fit_on_texts(X_text_train)
# Définition des dictionnaires
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

X_train = tokenizer.texts_to_sequences(X_text_train)
X_test = tokenizer.texts_to_sequences(X_text_test)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=len_max, padding='post')
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=len_max, padding='post')
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=len_max, padding='post', value=n_tags)
y_test = tf.keras.preprocessing.sequence.pad_sequences(y_test, maxlen=len_max, padding='post', value=n_tags)

print(X_train[:5,:], y_train[:5,:])
print('Shape of sentence X :', X_train.shape)
print('Shape of tags y :', y_train.shape)
print('Vocab size :', vocab_size)

