import requests as req
import json
import textwrap
import time

def get_genre(isbn):
     message = ""
     intento = 1
     while intento > 0:
          link = "https://www.googleapis.com/books/v1/volumes?q=isbn:"+ isbn
          obj = req.get(link).json()
          try:
               if 'error' in obj:
                    #print(f"error, esperemos unos segundos --> {isbn}")
                    time.sleep(0.5)
                    continue
               message = "getObject"
               categories = obj["items"][0]
               message = "getVolumeInfo"
               categories = categories["volumeInfo"]
               message = "getCategories"
               categories = categories["categories"]
               return categories
          except:
               # if message == "getObject":
               #      intento = intento - 1
               # else:
               #      intento = 0
               intento = 0
               print(f'{message} --> {link}')
               return None

