#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import fasttext

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,f1_score,recall_score


# In[8]:


# read in processed data
review_df = pd.read_csv("review_data.csv")
review_df = review_df[review_df.doc_len>0]
review_df = review_df[['review','label']]
review_df.head()


# In[9]:


# Obtain 70-30 train-test splits
X_train,X_test, y_train, y_test = train_test_split(review_df['review']
                                                  ,review_df['label']
                                                  ,train_size=0.7
                                                  ,test_size=0.3
                                                  ,random_state = 20191106)


# In[14]:

X_test = X_test.tolist()

review_df['formatted_text'] = '__label__'+review_df['label'].apply(str)+" "+review_df['review']
review_df[['formatted_text']].to_csv("fasttext_training.txt",index=False, header = False)


# In[13]:


review_df['formatted_text'].head()


# In[ ]:


# lr_lst = [0.05, 0.5, 1]
# dim_lst = [20, 40, 80]
# ws_lst = [3,5]

lr_lst = [0.05, 0.5, 1]
dim_lst = [20, 40, 80]
ws_lst = [3,5]

perf_lst = []

for lr in lr_lst:
    for dim in dim_lst:
        for ws in ws_lst:
            print('fitting for',lr,dim,ws)
            model = fasttext.train_supervised('fasttext_training.txt',lr=lr,dim=dim,ws=ws)
            y_pred_obj = model.predict(X_test)
            y_pred= [int(x[0][9]) for x in y_pred_obj[0]]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            perf_lst.append([lr, dim, ws, accuracy, precision, recall, f1])

                
perf_df = pd.DataFrame(perf_lst, columns = ['learning rate','vector dimension','window size','accuracy','precision','recall','f1'])  


# In[ ]:


perf_df.to_csv('perf_df_fasttext_2.csv', index=False)

