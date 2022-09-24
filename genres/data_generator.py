import numpy as np
import pandas as pd
import glob
import os
def make_dataset():
     if not os.path.exists('./resources/books_description.csv'):
          all_files = glob.glob(os.path.join('D:/PLN/original_dataset/libros_2M/', "*.csv"))

          df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
          df = df[['Name', 'Authors', 'ISBN', 'Rating', 'PublishYear','Description']]

          df = df.dropna(subset=['Name', 'ISBN'], how='all')
          df = df.dropna(subset=['Description'], how='all')
          df.to_csv('./resources/books_description.csv')

     if not os.path.exists('./resources/books_description_2.csv'):
          all_files = glob.glob(os.path.join('D:/PLN/original_dataset/books/', "*.csv"))

          df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
          df.rename(columns = {'book_authors':'Authors', 
                              'book_desc':'Description',
                              'book_isbn':'ISBN', 
                              'book_title':'Name',
                              'book_rating':'Rating', 
                              'genres':'Genres',
                              }, inplace = True)
          df = df[['Name', 'Authors', 'ISBN', 'Rating', 'Description','Genres']]
          df_aux = pd.read_csv('D:/PLN/original_dataset/archive/GoodReads_100k_books.csv')
          df_aux.rename(columns = {'author':'Authors', 
                              'desc':'Description',
                              'isbn':'ISBN', 
                              'title':'Name',
                              'rating':'Rating', 
                              'genre':'Genres',
                              }, inplace = True)
          df_aux = df_aux[['Name', 'Authors', 'ISBN', 'Rating', 'Description','Genres']]
          df = pd.concat([df,df_aux])
          df = df.dropna(subset=['Name', 'ISBN'], how='all')
          df = df.dropna(subset=['Description'], how='all')
          df.to_csv('./resources/books_description_2.csv')