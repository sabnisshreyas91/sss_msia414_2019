#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
import seaborn as sns
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd


# # Data Pre-process

# In[2]:


with open('D:\\Northwestern\\MSiA\\FQ 2019\\Text Analytics\\HW3\\reviews_Video_Games_5.json\\Video_Games_5.json') as handle:
    json_data = [[json.loads(line)['reviewText'],json.loads(line)['overall']] for line in handle]


# In[3]:


review_data = pd.DataFrame(json_data, columns = ['review','rating'])


# 1) Remove numbers and special characters from reviews <br>
# 2) Remove stop words <br>
# 3) Lemmatize

# In[4]:


def preprocess(row):
    file_str = row[0]
    # 1) remove numbers and spacial characters
    file_str = re.sub(r'([^a-zA-Z\s]+?)', ' ', file_str).replace("\n",' ')
    # 1) Lower case
    file_str = file_str.lower()
    
    #2) & 3) remove stop words and lemmatize
    file_str = ' '.join([lemmatizer.lemmatize(word) for word in file_str.split() if word not in stop_words])
review_data['review'] = review_data.apply(preprocess, axis =1)
review_data.head()


# In[51]:


sns.distplot(review_data.rating)


# We will treat any reviews <4 to be negative (0) and the rest to be positive (1)

# In[52]:


def label_pos_neg(row):
    if(row['rating']<4):
        return 0 
    else:
        return 1

review_data['label'] = review_data.apply(rating_pos_neg, axis=1)
review_data.head()


# # Summary stats

# In[66]:


num_docs = len(review_data)
review_data['doc_len'] = review_data['review'].str.len()
avg_doc_len = round(sum(review_data['doc_len'])/len(review_data))

print('Number of documents:',num_docs)
print('Average document length:',avg_doc_len)
ax = sns.distplot(review_data['doc_len']).set_title('Distribution of document length')
label_prop = round(100*review_data.label.value_counts()/len(review_data),2)
print('label distribution in %:')
print(label_prop)


# # Save processed data

# In[72]:


review_data.to_csv("review_data.csv", index=False)

