# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
from hazm import *
#import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.metrics import accuracy_score


files = [
    "noroz1.txt",
    "noroz2.txt",
    "noroz3.txt",
    "noroz4.txt",
    "noroz5.txt",
    "noroz6.txt",
    "sport.txt",
    "sport2.txt",
    "sport3.txt",
    "sport4.txt",
    "sport5.txt",
    "sport6.txt",
    "Geography1.txt",
    "Geography2.txt",
    "Geography3.txt",
    "Geography4.txt",
    "Geography5.txt",
    "Geography6.txt"    
    ]
# lable: 1: noroz , 2:sport ,3: Geography
lables = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
txts = []
mainAddress = "C:\\Users\\asma\\Corpuse\\"
for file in files:
    virtualAddress = mainAddress + file
    txt = open(virtualAddress, "r",encoding="utf-8" )
    txts.append(txt.read())
    #print(txt.read())
    #print("************")
    txt.close()
print(txts)
#Normalize:
for n in range(len(txts)):
     txts[n] = re.sub("\.|,|\?|،|\)|\(|:|;|-|–|؛|\u200c|\d|\[|]|\n|[a-z]|[A-Z]","",txts[n])

#print(txts)

txtAsWords = []   
for txt in txts:
    txtAsWords.append(word_tokenize(txt))
#print(txtAsWords)  

print("111#####################################")
for n in range(len(txtAsWords)):
    print(len(txtAsWords[n]))        


stopTxt = open(mainAddress+ "stop_words.txt","r",encoding="utf-8").read()

#print(stopWords.read())
stopWords = stopTxt.split("\n")
#print(stopWords)
for n in range(len(txtAsWords)):
    txtAsWords[n] = [w for w in txtAsWords[n] if(not (w in stopWords))]
    #for n in range(len(txt)):


stemmer = Stemmer()
lemmatizer = Lemmatizer()

for txt in txtAsWords:
    for n in range(len(txt)):
        print(txt[n]+":")
        txt[n] = stemmer.stem(txt[n])
        print(txt[n])
        #txt[n] = lemmatizer.lemmatize(s)
#        print(txt[n])
#print("2222#####################################")
# check that how much stopword is effect:
#for n in range(len(txtAsWords)):
#  print(len(txtAsWords[n]))      
#print(txtAsWords)

txts = []

for words in txtAsWords:
    #for n in range(len(txt)): 
    txt = ' '.join(words)
    #print(txt)
    txts.append(txt)
print(txts)
    
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(txts)
print(vectorizer.get_feature_names())

print("*****************")
print(X.toarray())

X = X.toarray()


import numpy as np


   
scaler = StandardScaler()
Xnormal=scaler.fit_transform(X)
y = lables
print(Xnormal)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=1) 
#random
#train the model
print("************************train")
#clf = LogisticRegression() #creat a L.R
clf = SGDClassifier()
print(clf) #print parameter of L.R
clf.fit(X_train, y_train) # train : fit the weight
#predict with Logistic Regression
y_pred = clf.predict(X_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))
