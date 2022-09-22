import requests as req
import json
import textwrap

def get_genre(isbn):
     link = "https://www.googleapis.com/books/v1/volumes?q=isbn:"+ str(isbn)
     obj = req.get(link).json()
     try:
          categories = obj["items"][0]["volumeInfo"]["categories"]
     except:
          print(link)
          return None

     return categories