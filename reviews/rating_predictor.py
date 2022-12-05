import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split

review_col = "review_text"
rating_col = "rating"
TEST_PERCENTAGE = 0.2

def rating_predictor(file):
     df = pd.read_csv(file)
     df = df[[rating_col,review_col]]
     df.dropna(inplace=True)
     #df = df.sample(500)
     df = df[(df[rating_col] != 0)]
     # df = df[(df[rating_col] != 0) & (df[rating_col] != 3)]
     # df[rating_col] = df[rating_col].apply(lambda rating : +1 if rating > 3 else -1)

     df = get_balanced(df,rating_col)
     
     #Split en test y train. Se considera balanceado para los generos
     X_train, X_test, y_train, y_test = train_test_split(df[review_col], df[rating_col] , test_size = TEST_PERCENTAGE, stratify=df[rating_col])
     
     # vectorizer = TfidfVectorizer(
     # sublinear_tf=True,
     # strip_accents='unicode',
     # analyzer='word',
     # token_pattern=r'\w{1,}',
     # stop_words=stopwords.words('english'),
     # ngram_range=(1, 1))

     # train_matrix = vectorizer.fit_transform(X_train)
     # test_matrix = vectorizer.transform(X_test)

     #Stack arrays in sequence horizontally (column wise).
     # This is equivalent to concatenation along the second axis, 
     # except for 1-D arrays where it concatenates along the first axis. 
     # Rebuilds arrays divided by hsplit.

     # train_features = hstack([train_matrix])
     # test_features = hstack([test_matrix])

     # classifier = LogisticRegression(solver='sag')
     # classifier.fit(train_features, y_train)
     # test_pred = classifier.predict(test_features)

     # print(confusion_matrix(y_pred = predictions,y_true = y_test))
     # print(accuracy_score(y_test,predictions))
     # print(sorted(df[rating_col].unique()))
     # plot_multivariable_confussion_matrix(y_test.to_numpy(),predictions,sorted(df[rating_col].unique()),"")
     print("------------------------------------------------------------------")
     
     vectorizer = CountVectorizer()
     train_matrix = vectorizer.fit_transform(X_train)
     test_matrix = vectorizer.transform(X_test)

     lr = LogisticRegression(max_iter=1000)
     lr.fit(train_matrix,y_train)
     predictions = lr.predict(test_matrix)
     print(confusion_matrix(y_pred = predictions,y_true = y_test))
     print(accuracy_score(y_test,predictions))
     print(sorted(df[rating_col].unique()))
     plot_multivariable_confussion_matrix(y_test.to_numpy(),predictions,sorted(df[rating_col].unique()),"")