from genres.data_generator import * 
from genres.data_manipulator import *
from reviews.data_manipulator import *
from genres.genre_predictor import *
from reviews.rating_predictor import *
from analizer import * 

GENERATE_DATASET = 0
CLEAN_DATASET = 1
ANALIZE_DATASET = 2
RESOLVE_DATASET = 3

state = RESOLVE_DATASET

book_path_source_1 = './resources/genres/books_description_2_clean.csv'
book_path_dest = './resources/genres/books_genres.csv'
#book_path_dest_aux = './resources/genres/books_genres_aux.csv'

book_path_source_2 = './resources/genres/books_description_clean_genres_backup.csv'
genre_path = './resources/genres.csv'

reviews_path_src = './resources/reviews/reviews_orig.csv'

reviews_path = './resources/reviews/reviews.csv'

if state == GENERATE_DATASET:
     make_dataset()
elif state == CLEAN_DATASET:
     clean_dataset(reviews_path_src,reviews_path)     
     clean_datasets()
     # descomentar si se quieren elimnar stopwords
     # clean_stopwords(book_path_dest,'Description',true,book_path_dest)
     
     search_genres(book_path_dest,genre_path,'genres')
     
     set_genres(book_path_dest,book_path_dest,'genres')    

elif state == ANALIZE_DATASET:
     analize_genres()
     analize_reviews()

elif state == RESOLVE_DATASET:
     predict_genres(book_path_dest)
     rating_predictor(reviews_path)