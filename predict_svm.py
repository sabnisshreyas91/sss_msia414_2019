#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import re
import nltk
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,f1_score,recall_score
import pickle
import argparse


# In[3]:


tfidf_vec = pickle.load(open('tfidf_vec_svm.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl','rb'))


# In[6]:


def preprocess(file_str):
    # 1) remove numbers and spacial characters
    file_str = re.sub(r'([^a-zA-Z\s]+?)', ' ', file_str).replace("\n",' ')
    # 1) Lower case
    file_str = file_str.lower()
    
    #2) & 3) remove stop words and lemmatize
    file_str = ' '.join([lemmatizer.lemmatize(word) for word in file_str.split() if word not in stop_words])
    return file_str


# In[7]:


def predict(text):
    text_lst = [text]
    test_obj = tfidf_vec.transform(text_lst)
    predict_val = model.predict(test_obj)
    predict_prob = model.predict_proba(test_obj)
    if predict_val[0] == 1:
        class_val = 'Positive'
        class_prob = predict_prob[0][1]
    else:
        class_val = 'Negative'
        class_prob = predict_prob[0][0]

    output_json = {}
    output_json['label'] = class_val
    output_json['probability'] = class_prob
    return output_json


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Review text")
    args = parser.parse_args()
    text = preprocess(args.text)
    output_json = predict(text)
    print(output_json)

