#Import packages
import pandas as pd
import numpy as np
import logging
from simpletransformers.classification import ClassificationModel
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
from nltk.tokenize import sent_tokenize, word_tokenize 
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('updated_train.csv')
df_test = pd.read_csv('updated_test.csv') #read dataset



def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)
    
  return input_txt


  
df_train['text'] = np.vectorize(remove_pattern)(df_train['text'], "@[\w]*") #remove @ and * from tweet
df_train['text'] = df_train['text'].str.replace("[^a-zA-Z#]", " ") # remove special characters, numbers, punctuations
df_test['text'] = np.vectorize(remove_pattern)(df_test['text'], "@[\w]*") #remove @ and * from tweet
df_test['text'] = df_test['text'].str.replace("[^a-zA-Z#]", " ") 



cv = CountVectorizer()
df_train['text'] = cv.fit_transform(df_train['text'])
X = df_train.drop(['target', 'ID'], axis= 1)
y = df_train.target




# TF-IDF feature matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df_train['text'])

# bag-of-words feature matrix
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df_train['text'])


# bag-of-words feature matrix
bow_vectorizer_test = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow_test = bow_vectorizer_test.fit_transform(x_test['text'])

# bag-of-words feature matrix
bow_tfidvectorizer = TfidfVectorizer(max_df=0.90, min_df=2, 
                                     max_features=1000, stop_words='english')
bow_tfid = bow_tfidvectorizer.fit_transform(df_train['text'])



xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, 
                                                          df_train['target'], random_state=42, test_size=0.3)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(xtrain_bow,ytrain)


y_pred = log_reg.predict(xvalid_bow)


log_loss(yvalid, y_pred, eps=1e-15)


