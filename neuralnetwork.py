from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils as ku
from collections import Counter
import numpy as np
import os
import nltk
import config

#nltk.download_shell() #IMPORTANT! Don't use nltk.download as it will crash macOS. Use the shell to download npkt packages

"""
Created on Mon Dec  10 15:00 2019

Objective: 
Create a hot encoded array from a given source of n size
Create a dataset from arrays from hot encoded arrays
Use it to train the network 

Now any given array from hot encoded arrays will be assigned a probability of 
belonging in a category based on automatically detected patterns

Tip: 
When overfitting, n might be too large
When underfitting, n might be too small

When adding a language do the following: 
    1. increase n_encoding with 10
    2. add syntax 
    3. train the network again

Note:   When this summary is placed above the keras imports it'll cause an error. 
        model_to_json seems to work when executed but on read it'll throw an ERROR: extra info  

@author: Stephen Goedhart
"""
n_categories = config.n_categories
n_words = config.n_words
n_input = config.n_input
n_layers = config.n_layers
n_encoding = config.n_encoding
    
# feed the network some common used words to speed up the process and make it more robust. 
#   Add language specific syntax/words when adding a language
syntax = ['let', 'var', 'struct', 'class', 'func', 'Int', 'Double', 'Float', 'String', 'Char', 'Void', 'typealias', '//', ':', '{', '}',
          'break', 'case', 'chan', 'const', 'continue', 'default', 'defer', 'else', 'fallthrough', 'for', 'func', 'Go', 'Goto', 'if', 
          'import', 'interface', 'map', 'package', 'range', 'return', 'select', 'Struct', 'Switch', 'Type', 'Var', 'fun', '/*', '*/', 
          'val', 'if', 'else', 'null', 'for', '(', ')', 'in', 'while', 'public:', 'private:', 'static', 'struct', 'fn', ';', 'println', 
          '::']

#load test data, clean it and store it
data = []
words = [] # shape = (3, n)
labels = []


# Get the data. Essential data can be found in config.py
for i in range(n_categories):
    files = os.listdir(config.file_dir + config.categories[i] + "/")
    words.append(i)
    words[i] = []
    for j in range(len(files)):
        file_source = open(config.file_dir + config.categories[i] + "/" + files[j])
        file_content =  file_source.read()
        file_content_tokenized = nltk.word_tokenize(file_content) #flat list of words
        file_source.close()
        
        for w in file_content_tokenized:
            words[i].append(w)
            
        data.append(file_content_tokenized)
        labels.append(i)
    
# flatten list and create a list with unique values so that it's ready for one hot encoding              
for i in range(len(words)):
    common = Counter(words[i]).most_common(n_words)
    words[i] = []
    for c in common:
        words[i].append(c[0])  
        
unique_list = []
for w in syntax:
    if w not in unique_list:
        unique_list.append(w)
        
for i in range(n_categories):
    for w in words[i]:
        word = w
        if w not in unique_list:
            unique_list.append(word)          
dictionary = unique_list[:n_encoding]

#make sure that every entry has enough characters 
encoded_data = []
prepare_labels = []
for i in range(len(data)):
    data[i] = config.one_hot_encode(data[i], dictionary) # array of one hot encoded words
    iterator = int(len(data[i]) / n_input)
    for j in range(iterator):
        start = 0 + j * n_input
        snippet = data[i][start : start + n_input]
        prepare_labels.append(labels[i])
        encoded_data.append(snippet)
        
one_hot_labels = ku.to_categorical(prepare_labels, num_classes=n_categories)

#flatten the data (it's a 3d array now. We might wanna skip this step when we use an LSTM)
encoded_sentence = []
for x in encoded_data:
    sublist = []
    for y in x:
        sublist += y
    encoded_sentence.append(sublist)
encoded_data = np.array(encoded_sentence)

# create, compile and train the model
model = Sequential([
        Dense(n_layers, input_dim =n_input * n_encoding), #input dimensions are equal to amount of characters (integers) per sample
        Activation('relu'), #activation function chosen by trial and error
        Dense(n_categories), 
        Activation('softmax'),
        ])

# Compile the model. Adam chosen by default
model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Train the model, aim for an accuracy rate of 98%
model.fit(encoded_data, one_hot_labels, epochs=10000, batch_size=n_input * n_encoding, verbose=1) 

# save the model and the dictionary
json_model = model.to_json()
json_file = open('./model.json', 'w')
json_file.write(json_model)
json_file.close()
model.save_weights('weight.h5')

dictionary_file = open('./dictionary.txt', 'w')
dictionary_file.write(str(','.join(dictionary)))
dictionary_file.close()

print('code: 200')