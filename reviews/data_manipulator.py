from asyncio.windows_events import NULL
from cmath import nan
import numpy as np
import pandas as pd
import glob
import os
from utils import *
import time
from datetime import timedelta

def clean_data(data):
  #Remove HTML
  data = re.sub(r'<.*?>', '',  data)
  #Remove URL
  data = re.sub(r'http\S+', '', data)
  #Remove punctuation
  data = re.sub(r'[^\w\s]', '', data)
  #Remove numbers
  data = re.sub(r'\b\d+\b', '', data)
  #Create a list os emojis
  emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
  #Remove emojis
  data = emoji_pattern.sub(r'', data)
  #Make all letters lower case
  data = data.lower()
  return data

def clean_dataset(file_src,file_dest):
     df_reviews = pd.read_csv(file_src)
     df_reviews = delete_invalid_format(df_reviews,'review_text',True)
     print("Format and Language done... 1")
     df_reviews = get_lang(df_reviews)
     print("get lang done... 1")
     df_reviews['review_text'] = df_reviews['review_text'].apply(lambda z: clean_data(z))
     df_reviews = cleaner(df_reviews,'review_text')
     print("cleaner title done... 1")
     df_reviews.to_csv(file_dest)
