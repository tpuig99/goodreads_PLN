from asyncio.windows_events import NULL
from cmath import nan
import numpy as np
import pandas as pd
import glob
import os
from utils import *
from genreFetcher import * 
import time
from datetime import timedelta

def clean_datasets():
     if not os.path.exists('./resources/books_description_2_clean.csv'):
          df_genres = pd.read_csv('./resources/books_description_2.csv')
          df_genres = delete_invalid_format(df_genres,True)
          print("Format and Language done... 1")
          df_genres = get_lang(df_genres)
          print("get lang done... 1")
          df_genres = cleaner(df_genres)
          print("cleaner done... 1")
          df_genres.to_csv('./resources/books_description_2_clean.csv')

     if not os.path.exists('./resources/books_description_clean.csv'):
          df_books = pd.read_csv('./resources/books_description.csv')
          df_books = delete_invalid_format(df_books,True)
          print("Format and Language done... 2")
          df_books = get_lang(df_books)
          print("get lang done... 2")
          df_books = cleaner(df_books)
          print("cleaner done... 2")

          df_books.to_csv('./resources/books_description_clean.csv')

def get_best_genres(plot=False):
     if not os.path.exists('./resources/genres.csv'):
          df_genres = pd.read_csv('./resources/books_description_2_clean.csv')
          df_genres = df_genres.dropna(subset=['Genres'])
          df_genres['Genres'] = df_genres['Genres'].str.lower() 
          items = df_genres['Genres'].to_list()
          genres_list = []
          genres = set()
          a = 0
          for item in items:
               item = item.lower()
               genre = re.split(r"[,|]", item)
               genres.update(set(genre))
               genres_list += genre
          
          if plot:
               bar(pd.DataFrame(genres_list,columns=['Genres']),'Genres')
          genres = pd.DataFrame(genres_list,columns=['Genres'])
          groupedDF = genres.value_counts()[:100].to_frame(name='count')
          groupedDF.reset_index(inplace=True) 
          groupedDF.to_csv('./resources/genres.csv')

def get_legal_genres(item):
     item = item[0].lower()

     genres = pd.read_csv('./resources/genres.csv')
     #primero chequeo por los titulares 
     for g in genres['Genres']:
          if item in (g) or g in item:
               return g

     for _,genre in genres.iterrows():
          list = re.split(r"[,|]", genre['Similar'])
          for g in list:
               if item in (g) or g in item:
                    return genre['Genres']
     return None

def set_genres():
     df_genres = pd.read_csv('./resources/books_description_2_clean.csv')
     df_genres = df_genres.dropna(subset=['Genres'])
     
     for index,item in df_genres.iterrows():
          item = item['Genres'].lower()
          genre = re.split(r"[,|]", item,1)[0]
          df_genres.loc[index, 'Genre1'] = genre 
     df_genres.to_csv('./resources/books_description_2_clean.csv')
     
     genres = pd.read_csv('./resources/genres.csv')

     for index,item in df_genres.iterrows():
          for _,genre in genres.iterrows():
               list = re.split(r"[,|]", genre['Similar'])
               if item['Genre1'] in (list):
                    df_genres.loc[index, 'Genre1'] = genre['Genres']
                    break      

def search_genres():
     start_time = time.time()
     df_books = pd.read_csv('./resources/books_description_clean_genres.csv', index_col=0)
     df_auxi = pd.read_csv('./resources/books_description_clean_genres_aux.csv', index_col=0)
     not_found = 0
     for idx,book in df_books.iterrows():
          if idx < 13000:
               book_auxi = df_auxi[(df_auxi['Name'].eq(book['Name']))]
               if book_auxi.shape[0] != 0 and book_auxi['Genre1'].any():
                    df_books.loc[idx, 'Genre'] = book_auxi['Genre']
                    df_books.loc[idx, 'Genre1'] = book_auxi['Genre1']
                    continue
          isbn = book['ISBN'].zfill(10)

          genre = get_genre(isbn)
          if genre is not None:
               genre1 = get_legal_genres(genre)
               df_books.loc[idx, 'Genre'] = genre
               df_books.loc[idx, 'Genre1'] = genre1
          else:
               #df_books.drop(index, inplace=True)
               print(book['Name'])
               not_found += 1
          if idx % 1000 == 0:
               df_books.to_csv(path_or_buf='./resources/books_description_clean_genres.csv', index=False)
               print(f'{idx} - not found {not_found} - time: {str(timedelta(seconds=time.time() - start_time))}')
