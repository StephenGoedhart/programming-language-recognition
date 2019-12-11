#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:55:28 2019

To do:
    1:  Devide the file in as many sentences of 12 words as possible.
        summerize predictions per language
        choose the language with the highest prediction
    

@author: stephen
"""

from keras.models import model_from_json
import numpy as np
import os
import config as c
import nltk

n_categories = c.n_categories
n_words = c.n_words
n_input = c.n_input
n_layers = c.n_layers
n_encoding = c.n_encoding

#the model to be predicted
predict_file = open('./predict/sample.txt')
predict_string = predict_file.read()
predict_tokenized = nltk.word_tokenize(predict_string)
predict_file.close()


dictionary_file = open('dictionary.txt')
dictionary = dictionary_file.read()
dictionary = dictionary.split(",")
dictionary_file.close()


encoded_prediction = c.one_hot_encode(predict_tokenized, dictionary)[:n_input]

p = []
#flatten the data array
for w in encoded_prediction:
    p += w

encoded_prediction = p
encoded_prediction = np.array([encoded_prediction])

json_file =  open('model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('weight.h5')

prediction = model.predict(encoded_prediction, verbose=0)
print(c.get_category(prediction))
