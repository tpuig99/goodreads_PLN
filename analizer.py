from cmath import nan
import pandas as pd
from utils import *
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def analize_genres():
     df_genres = pd.read_csv('./resources/genres/books_genres_finale.csv')
     bar(pd.DataFrame(df_genres,columns=['Genre']),'Genre','Generos')
     nltk.download('stopwords')
     stopw = set(stopwords.words('english')) 
     stopw.update(['book','story','new','use','using','one','author'])
     for genre in df_genres['Genre'].unique():
          df_aux = df_genres[df_genres['Genre'] == genre]
          text = " ".join(i for i in df_aux.Description)
          wordcloud = WordCloud(stopwords=stopw).generate(text)
          plt.imshow(wordcloud, interpolation='bilinear')
          plt.title(f"Wordcloud for {genre}")
          plt.axis("off")
          plt.show()

def analize_reviews():
     df_reviews = pd.read_csv('./resources/reviews/reviews_clean.csv')
     
     bar(df_reviews,'rating',"Ratings")
     bar(df_reviews,'Language',"Language")
     nltk.download('stopwords')
     stopw = set(stopwords.words('english')) 
     stopw.update(['book','story','new','use','using','one','author','read','reading','like','character','know',
     'think','would','get','even','could','thing','make','also','books','going','though'])
     for rating in range(0,6):
          df_rating = df_reviews[df_reviews['rating'] == rating]
          text = " ".join(i for i in df_rating.review_text)
          wordcloud = WordCloud(stopwords=stopw).generate(text)
          plt.imshow(wordcloud, interpolation='bilinear')
          plt.title(f"Wordcloud for rating {rating}")
          plt.axis("off")
          plt.show()
          print(f"\n\n\n\n{rating}\n")
          tfidf = checktfidf(df_rating)
          print(tfidf.head(15))

def checktfidf(df_reviews):
     #TF-IDF gives high scores to terms occurring in only very few documents, 
     # and low scores for terms occurring in many documents, 
     # so its roughly speaking a measure of how discriminative a term is in a given document.
     tfIdfVectorizer = TfidfVectorizer(ngram_range=[1,1], max_df=0.8, min_df=2, max_features=None, stop_words="english")
     tfIdf = tfIdfVectorizer.fit_transform(df_reviews['review_text'])
     df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
     df = df.sort_values('TF-IDF', ascending=False)
     return df
