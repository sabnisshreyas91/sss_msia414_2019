#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,f1_score,recall_score


# In[20]:


# read in processed data
review_df = pd.read_csv("review_data.csv")
review_df = review_df[review_df.doc_len>0]
review_df = review_df[['review','label']]
review_df.head()


# In[34]:


# Obtain 70-30 train-test splits
X_train,X_test, y_train, y_test = train_test_split(review_df['review']
                                                  ,review_df['label']
                                                  ,train_size=0.7
                                                  ,test_size=0.3
                                                  ,random_state = 20191106)


# In[56]:


# define grid of hyperparameters
ngram_lst = [(1,1),(2,2)]
min_tf_lst = [0.1,0.15, 0.2]

reg_lst = ['l2','l1']
reg_str_lst = [0.1, 1, 10]

perf_lst= []

for ngram in ngram_lst:
    for min_tf in min_tf_lst:
        
        tfidf_vec = TfidfVectorizer(ngram_range = ngram,min_df = min_tf).fit(X_train)
        training_obj = tfidf_vec.transform(X_train)
        
        for reg in reg_lst:
            for reg_str in reg_str_lst:
                print('fitting for:',ngram, min_tf, reg, reg_str)
                model = LogisticRegression(penalty=reg, C=reg_str)
                model.fit(training_obj,y_train)
                testing_obj = tfidf_vec.transform(X_test)
                y_pred = model.predict(testing_obj) 
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
    
                perf_lst.append([ngram, min_tf, reg, reg_str, accuracy, precision, recall, f1])

perf_df = pd.DataFrame(perf_lst, columns = ['ngram','min_tf','regularization type','regularization strength','accuracy','precision','recall','f1'])      


# In[60]:


perf_df.to_csv("logistic_regression_perf.csv", index=False)
perf_df.sort_values('f1', ascending=False)

