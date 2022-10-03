from asyncio.windows_events import NULL
from cmath import nan
import numpy as np
import pandas as pd
import glob
import os
from utils import *
from genres.genreFetcher import * 
from datetime import timedelta
from genres.genre_completer import *

def clean_datasets():
     if not os.path.exists('./resources/books_description_2_clean.csv'):
          df_genres = pd.read_csv('./resources/books_description_2.csv')
          df_genres = delete_invalid_format(df_genres,'Description',True)
          print("Format and Language done... 1")
          df_genres = get_lang(df_genres)
          print("get lang done... 1")
          df_genres = cleaner(df_genres,'Description')
          print("cleaner description done... 1")
          df_genres = cleaner(df_genres,'Name',True)
          print("cleaner title done... 1")
          df_genres.to_csv('./resources/books_description_2_clean.csv')

     if not os.path.exists('./resources/books_description_clean.csv'):
          df_books = pd.read_csv('./resources/books_description.csv')
          df_books = delete_invalid_format(df_books,'Description',True)
          print("Format and Language done... 2")
          df_books = get_lang(df_books)
          print("get lang done... 2")
          df_books = cleaner(df_books,'Description')
          print("cleaner done... 2")
          df_books = cleaner(df_books,'Name',True)
          print("cleaner title done... 1")

          df_books.to_csv('./resources/books_description_clean.csv')

#FOR FILE WHERE GENRES DOESNT EXIST
def search_genres(file_orig,temp_file_dest,genre_column,init,state=False):
     df_books = pd.read_csv(file_orig)
     batch = 2000
     for idx,book in df_books[init:].iterrows():
          if pd.notnull(book[genre_column]):
               if idx!= 0 and idx % batch == 0:
                    auxi_file_writer(temp_file_dest,df_books,idx,batch,state)
               if state:
                    state = False
               continue
          if pd.isnull(book[genre_column]):
               isbn = str(book['ISBN']).zfill(10)
               genre = get_genre(isbn)
          else:
               genre = book[genre_column]
          
          genre = clean_text(genre)
          genre_legal = get_legal_genres(genre)
          if genre_legal is not None:
               df_books.loc[idx, genre_column] = genre_legal
          else:
               df_books.drop(idx, inplace=True)

          if idx!= 0 and idx % batch == 0:
               auxi_file_writer(temp_file_dest,df_books,idx,batch,state)
               if state:
                    state = False

#for file with genres
#From each book get first genre of list as the representative
#If that genre it is not on de general genres list it adds it to another csv file
def set_genres(books_file,dest_file,genre_col):
     df_books = pd.read_csv(books_file)
     df_books = df_books.dropna(subset=[genre_col])
     genres = pd.read_csv('./resources/genres.csv')
     
     #get first genre of list as representative genre
     for index,item in df_books.iterrows():
          item = item[genre_col].lower()
          genre = re.split(r"[,|]", item,1)[0]
          genre_legal = get_legal_genres(genre)
          if genre_legal is not None:
               df_books.loc[index, 'Genre1'] = genre_legal
          else:
               df_books.drop(index, inplace=True) 
     df_books.to_csv(dest_file)
     
     

           