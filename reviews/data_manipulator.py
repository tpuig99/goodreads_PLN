from asyncio.windows_events import NULL
from cmath import nan
import numpy as np
import pandas as pd
import glob
import os
from utils import *
import time
from datetime import timedelta

def clean_dataset():
     if not os.path.exists('./resources/reviews/reviews_clean.csv'):
          df_reviews = pd.read_csv('./resources/reviews/reviews.csv')
          df_reviews = delete_invalid_format(df_reviews,'review_text',True)
          print("Format and Language done... 1")
          df_reviews = get_lang(df_reviews)
          print("get lang done... 1")
          df_reviews = cleaner(df_reviews,'review_text')
          print("cleaner title done... 1")
          df_reviews.to_csv('./resources/reviews/reviews_clean.csv')
