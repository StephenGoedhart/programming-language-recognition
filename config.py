#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:52:27 2019

Change settings used by the network so that they're automatically used for predict.py

@author: stephen
"""

import os

file_dir = "./data/"
categories = os.listdir(file_dir)
blacklist = ['.DS_Store']

for i in range(0, len(blacklist)):
    if(categories.count(blacklist[i]) > 0):
        categories.remove(blacklist[i])

n_categories = len(categories)
n_words = 50 # number of most common words per category
n_input = 12 # 12 words per sentence
n_layers = 4 # amount of layers in network
n_encoding = 30 # length of one hot encoding

# some custom functions

# Encode an array as an one hot encoded version given a dictionary.
#  if it doesn't exist in the dict, nothing will be returned

def one_hot_encode(arr, dic):
    encoding = []
    for w in arr:
        if(dic.count(w) > 0):
            index = dic.index(w)
            encoded_array = []
            encoded_array += [0] * index
            encoded_array += [1]
            encoded_array += [0] * (len(dic) - index - 1)
            encoding.append(encoded_array)
    return encoding

# return the category that had the highest probability
def get_category(cats):
    res = 0
    cat = 0
    for i in range(len(cats[0])):
        if(cats[0,i] > res):
            res = cats[0,i]
            cat = i
    return categories[cat]
