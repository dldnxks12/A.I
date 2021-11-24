import nltk
import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt

import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Dataset
pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('./data/movie_data.csv', sep = ',')

# Reduce Size for Memory Capacity
data = data.sample(frac = 1)
data = data[:5000]

print("Check Unique Labels : ", np.unique(data['sentiment']))

#Processing
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.WordNetLemmatizer()

def clean_text(text):
    # Remove Punctuation
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    # make Token
    tokens = re.split('\W+', text)
    # Remove StopWords and make Stem
    text = [ps.lemmatize(word) for word in tokens if word not in stopwords]
    return text


# Vectorization -- TF-IDF 사용해보자
TFIDF_Vect = TfidfVectorizer(analyzer = clean_text)
X_TFIDF = TFIDF_Vect.fit_transform(data['review'])
X_TFIDF_pd = pd.DataFrame(X_TFIDF.toarray(), columns = TFIDF_Vect.get_feature_names())

# Model Selection  --- SVM / Gradient Boosting Trees, Random Forest
warnings.filterwarnings('ignore', category = DeprecationWarning)

rf = RandomForestClassifier()
param = {'n_estimators' : [10, 150, 300],
         'max_depth' : [30, 60, 90, None]}

gs = GridSearchCV(rf, param, cv = 5, n_jobs = 4)
gs_fit = gs.fit(X_TFIDF_pd, data['sentiment'])
print(pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending = False).head())

