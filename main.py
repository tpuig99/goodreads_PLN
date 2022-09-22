from data_generator import * 
from data_manipulator import *

from genreFetcher import * 

# make_dataset()
# clean_datasets()
# get_best_genres()
# set_genres()

search_genres(18000)

# df_books = pd.read_csv('./resources/books_description_clean_genres.csv', index_col=0)
# book = df_books.iloc[[18129]]
# print(book)
# isbn = str(book['ISBN'].values[0]).zfill(10)
# print(f'|{isbn}|')
# print(get_genre(isbn))