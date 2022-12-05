from cmath import nan
import numpy as np
import pandas as pd
import re 
from langdetect import detect
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
stop = stopwords.words('english')

#If second param true, adds column language
def delete_invalid_format(df,column,lang=False):
     indx=[]
     for i in df.index:
          try:
               language = detect(df.at[i,column])
               if lang:
                    df.loc[i, 'Language'] = language 
          except:
               indx.append(i)
     print(len(indx))
     df=df.drop(index=indx)
     return df

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
     if text is None:
          return None
     if isinstance(text,list):
          text = text[0]
     text = text.lower()
     text = " ".join(text.split())
     text = re.sub(re.compile('<.*?>'), '', text) #delete tags <br>
     text = re.sub(r"what's", "what is ", text)
     text = text.replace('(ap)', '')
     text = re.sub(r"\'s", " is ", text)
     text = re.sub(r"\'ve", " have ", text)
     text = re.sub(r"can't", "cannot ", text)
     text = re.sub(r"n't", " not ", text)
     text = re.sub(r"i'm", "i am ", text)
     text = re.sub(r"\'re", " are ", text)
     text = re.sub(r"\'d", " would ", text)
     text = re.sub(r"\'ll", " will ", text)
     text = re.sub(r'\W+', ' ', text)
     text = re.sub(r'\s+', ' ', text)
     text = re.sub(r"\\", "", text)
     text = re.sub(r"\'", "", text)    
     text = re.sub(r"\"", "", text)
     text = re.sub('[^a-zA-Z ?! 0-9]+', '', text)
     text = _removeNonAscii(text)
     text = text.strip()
     return text

def cleaner(df,column,duplicates=False):
     df[column] = df[column].apply(clean_text)
     df = df[(pd.notnull(df[column]))&(df[column].ne(''))]
     if duplicates:
          df = df.drop_duplicates(column)
     return df

lang_lookup = pd.read_html('https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes')[0]
langpd = lang_lookup[['ISO language name','639-1']]
langpd.columns = ['language','iso']

def desc_lang(x):
     if x in list(langpd['iso']):
          return langpd[langpd['iso'] == x]['language'].values[0]
     else:
          return 'nil'

def get_lang(df):
     df['Language'] = df['Language'].apply(desc_lang)
     return df

def bar(df, var,title=''):
    grouped = df.groupby(var, dropna=False)
    names = []
    count = []
    for name,group in grouped:
          names.append(name)
          count.append(group.shape[0])
     
    fig = px.bar(df, x=names, y=count, text_auto=True)
    fig.update_layout(
        title=f"{var} - {title}", 
        yaxis_title="Cantidad", 
        font={'size': 18} 
    )    
    fig.show()


def clean_stopwords(df,column,save=False,path=""):
     df[column] = df[column].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
     if save:
          df.to_csv(path)
     return df
     
def get_balanced(df,col):
     max_per_group = min(df[col].value_counts())
     df = df.groupby(col, group_keys=False).apply(lambda x: x.sample(max_per_group))
     return df

def plot_multivariable_confussion_matrix(y_true,y_predict,labels,title):
    matrix = np.zeros((len(labels),len(labels)))
   
    for i in range(1,len(y_true)):
        real_index = np.where(labels == y_true[i])[0][0]
        predict_index = np.where(labels == y_predict[i])[0][0]
        matrix[real_index][predict_index] += 1
    
    fig = px.imshow(matrix, text_auto=True, x=labels,y=labels,color_continuous_scale='purpor')

    fig.update_layout(
        title=f"{title}",
        xaxis_title="Predicción",
        yaxis_title="Real", 
        font= { 'size': 18 }
    )
    fig.show()
