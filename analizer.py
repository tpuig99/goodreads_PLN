from asyncio.windows_events import NULL
from cmath import nan
import numpy as np
import pandas as pd
import glob
import os
from utils import *
import time
from datetime import timedelta
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def analize_genres():
     df_genres = pd.read_csv('./resources/books_description_clean_genres_backup.csv')
     bar(pd.DataFrame(df_genres,columns=['Genres1']),'Genres1','Generos')

def analize_reviews():
     df_reviews = pd.read_csv('./resources/reviews/reviews_clean.csv')
     
     bar(df_reviews,'rating',"Ratings")
     bar(df_reviews,'Language',"Language")
     
     nltk.download('stopwords')
     stopw = set(stopwords.words('english')) 
     for rating in range(0,6):
          df_rating = df_reviews[df_reviews['rating'] == rating]
          text = " ".join(i for i in df_rating.review_text)
          wordcloud = WordCloud(stopwords=stopw).generate(text)
          plt.imshow(wordcloud, interpolation='bilinear')
          plt.title(f"Wordcloud for rating {rating}")
          plt.axis("off")
          plt.show()