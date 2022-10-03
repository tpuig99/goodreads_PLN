import numpy as np
import pandas as pd
import glob
import os
from utils import *
from genres.genreFetcher import * 
from datetime import timedelta

##AUX FILE TO ALL GENRE GET FUNCTION 

#From file get first 50 genres as the general genres of all books
def get_best_genres(file_orig,genre_col,plot=False):
     if not os.path.exists('./resources/genres.csv'):
          df_genres = pd.read_csv(file_orig)
          df_genres = df_genres.dropna(subset=[genre_col])
          df_genres[genre_col] = df_genres['Genres'].str.lower() 
          items = df_genres[genre_col].to_list()
          genres_list = []
          genres = set()
          a = 0
          for item in items:
               item = item.lower()
               genre = re.split(r"[,|]", item)
               genres.update(set(genre))
               genres_list += genre
          
          if plot:
               bar(pd.DataFrame(genres_list,columns=[genre_col]),'Genres')
          genres = pd.DataFrame(genres_list,columns=[genre_col])
          groupedDF = genres.value_counts()[:100].to_frame(name='count')
          groupedDF.reset_index(inplace=True) 
          groupedDF.to_csv('./resources/genres.csv')

#From genre gets its general name
def get_legal_genres(item):
     if item is None:
          return None
     if isinstance(item,list):
          item = item[0]
     
     item = item.lower()
     genres = pd.read_csv('./resources/genres.csv')
     #primero chequeo por los titulares 
     for g in genres['Genres']:
          if item in (g) or g in item:
               return g

     for _,genre in genres.iterrows():
          glist = re.split(r"[,|]", genre['Similar'])
          for g in glist:
               if item in (g) or g in item:
                    return genre['Genres']
     return None

def auxi_file_writer(temp_file_dest,df_books,idx,batch,state):
     df_books[idx-batch+1:idx].to_csv(path_or_buf=temp_file_dest,mode='a',header=state)

def get_new_genres():
     df_books = pd.read_csv('./resources/books_description_clean_genres.csv', index_col=0)
     df_books = df_books[(pd.notnull(df_books['Genre']))&(df_books['Genre'].ne(''))&(pd.isnull(df_books['Genre1']))]
     df_books = pd.DataFrame(df_books['Genre'].unique(),columns=["Genre"])
     df_books = cleaner(df_books,'Genre')
     df_books.to_csv('./resources/Genre1.csv')
