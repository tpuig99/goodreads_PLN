from cmath import nan
import pandas as pd
from utils import *
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models, layers
from tensorflow.keras.layers import LSTM,Dropout,Embedding
from sklearn.metrics import confusion_matrix,classification_report


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

description_col = "Description"
title_col = "Name"
genre_col = "Genre"

MAX_LENGHT = 150 #tras analisis: for 165 with 267528 are 250406--> 0.9359992225112885
TEST_PERCENTAGE = 0.2

def description_length_analisis(df):
     #df = pd.read_csv('./resources/genres/books_genres_finale.csv')
     df['description_word_count']  = df[description_col].str.split().str.len()
     grouped = df.groupby('description_word_count', dropna=False)
     
     names = []
     count = []
     for name,group in grouped:
               names.append(name)
               count.append(group.shape[0])
          
     fig = px.bar(df, x=names, y=count, text_auto=True)
     fig.update_layout(
          title=f"Description Word Count", 
          yaxis_title="Cantidad", 
          font={'size': 18} 
     )    
     fig.show()

     len_df=df['description_word_count'].value_counts(bins=100, normalize=True).reset_index().sort_values(by=['index'])
     len_df['cumulative']=len_df['description_word_count'].cumsum()
     len_df['index']=len_df['index'].astype('str')
     px.bar(len_df, x="index", y="cumulative", text_auto=True).show()

     total_reg = df.shape[0]
     
     max_lenght = 165 #for 165 with 267528 are 250406--> 0.9359992225112885
     part_reg_size = 0
     for name,group in grouped:
          if name <= max_lenght:
               part_reg_size += group.shape[0]
     
     print(f"for {max_lenght} with {total_reg} are {part_reg_size}--> {part_reg_size/total_reg}")

def tokenize_data(df):
     descriptions = df[description_col]

     # Tokenize our training data
     tokenizer = Tokenizer()
     
     tokenizer.fit_on_texts(descriptions)

     # Get our training data token word
     token_word = tokenizer.index_word
     
     # Get our training data word token
     word_token = tokenizer.word_index

     # Encode training data sentences into sequences
     train_sequences = tokenizer.texts_to_sequences(descriptions)

     # Pad the training sequences
     train_padded = pad_sequences(train_sequences, padding='pre', truncating='post', maxlen=MAX_LENGHT)

     # Output the results of our work
     # print(descriptions)
     # print("Word token:\n", word_token)
     # print("Token Word:\n", token_word)
     # print("\nPadded training sequences:\n", train_padded)

     return token_word,word_token,train_padded

def define_model(x_train, y_train, x_val, y_val,vocab_size,categories_count):
     model = models.Sequential()
     act = 'relu'
     model.add(layers.Dense(int(MAX_LENGHT * 0.8), activation=act, input_shape=(MAX_LENGHT,))) 
     # model.add(layers.Dense(int(MAX_LENGHT * 0.6), activation=act)) 
     model.add(layers.Dense(int(MAX_LENGHT * 0.4), activation=act))
     # model.add(layers.Dense(int(MAX_LENGHT * 0.2), activation=act))
     model.add(layers.Dense(int(MAX_LENGHT * 0.1), activation=act))

     model.add(layers.Dense(categories_count, activation='softmax'))

     model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['categorical_accuracy'])

     model.fit(x_train, y_train, 
                      epochs=100
                     #validation_data=(x_val, y_val)
                     )
     return model

def define_model_aux(x_train, y_train, x_val, y_val,vocab_size,categories_count):
     model = models.Sequential()
     model.add(layers.Embedding(input_dim=vocab_size, 
                              output_dim=50, 
                              input_length=MAX_LENGHT))
     model.add(layers.Flatten())
     model.add(layers.Dense(10, activation='relu'))
     

     model.add(layers.Dense(categories_count, activation='softmax'))

     model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['categorical_accuracy'])

     model.fit(x_train, y_train, 
                      epochs=15
                     #validation_data=(x_val, y_val)
                     )
     return model


def predict_genres(file):
     df = pd.read_csv(file,index_col=0)
     # 'academic','art','business','childrens','classics',
     # 'computer science','family','fantasy','fiction','food and drink',
     # 'historical','horror','literature','mystery','nonfiction','philosophy','poetry',
     # 'psychology','religion','romance','science','science fiction','spiritual','sports','young adult'
     df = df[df[genre_col].isin(['academic','art','business','childrens','classics',
     'computer science','family','fantasy','fiction','food and drink',
     'historical','horror','literature','mystery','nonfiction','philosophy','poetry',
     'psychology','religion','romance','science','science fiction','spiritual','sports','young adult'])]
     #df = df.sample(2000)
     
     df = get_balanced(df,genre_col)
     #eliminamos desc menores a 5 palabras
     df=df[( df[description_col].str.len() > 5 )].reset_index(drop=True)
     
     #Analisis para decidir la cantidad de palabras por descripcion
     #description_length_analisis(df)
     #Necesitamos tokenizar las descripciones para keras
     token_word,word_token,description_token = tokenize_data(df)
     df['description_token'] = description_token.tolist()

     #Vamos a transformar las etiquetas de números enteros a un vector binario (one-hot)
     encoder = LabelBinarizer()
     genre_categorical = encoder.fit_transform(df[genre_col])
     categories_count = len(genre_categorical[0])
     if categories_count == 1:
          genre_categorical = np.hstack((genre_categorical, 1 - genre_categorical))
          categories_count = len(genre_categorical[0])

     df['genre_categorical'] = genre_categorical.tolist()

     #Split en test y train. Se considera balanceado para los generos
     X_train, X_test, y_train, y_test = train_test_split(description_token, genre_categorical , test_size = TEST_PERCENTAGE, stratify=genre_categorical)


     #Agarramos del conjunto de training un conjunto de validación
     X_val=[] 
     y_val = []
     #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1,stratify=y_train)
    
     model = define_model_aux(X_train, y_train, X_val, y_val,len(token_word)+1,categories_count)

     test_loss, test_accuracy = model.evaluate(X_test, y_test)
     y_predict = model.predict(X_test)
     y_pred_lab = encoder.inverse_transform(y_predict)
     y_test_lab = encoder.inverse_transform(y_test)

     
     print(f"\n\n\n results")
     print(f"\nAccuracy: {test_accuracy}\n\n{df[genre_col].unique()}")
     print(confusion_matrix(y_pred = y_pred_lab,y_true = y_test_lab))
     plot_multivariable_confussion_matrix(y_test_lab,y_pred_lab,df[genre_col].unique(),"")
    
