import streamlit as st
import pickle 
import pandas as pd
import re  
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
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(min_df = 0.1, max_df = 0.9)

model = pickle.load(open('model.pkl', 'rb'))


st.cache()

def data_pro(text):
    df = {'text':[text]}
    df = pd.DataFrame(df)
    #return df

#df = data_pro(text)

tokens = []
def num_token(df):
    for line in df.text:
        token = nltk.word_tokenize(line)
        tokens.append(len(token))
        df['No_Token'] = pd.Series(tokens)

def split(text):
  return text

#num_token(df)

charater = []
def char(df):
    for x in df.text.iloc[0:]:
        chara = split(x)
        charater.append(len(chara))
        df['No_Characters'] = charater
        
#char(df)

#df = df.drop(['text'], axis = 1)

def predict(df):
    prediction = model.predict(df)
    return prediction

#"""if output == 1:
# st.markdown('Covid related tweet')
# else:
# st.markdown('This tweet is not covid related')"""


st.title('Covid text detection app')


text = st.text_input("Text")
df = {'text':[text]}
df = pd.DataFrame(df)
num_token(df)
char(df)
df = df.drop(['text'], axis = 1)



#input_df =pd.Dataframe(input)

if st.button("Classify"):
    output = predict(df = df)
    #st.success("This text or tweet is Covid {}".format(output))

    if output == 1:
        st.markdown('Your tweet is covid related')
    else:
        st.markdown('Your tweet is not covid related')



