import nltk
import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt

import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Dataset
pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('./data/movie_data.csv', sep = ',')

data = data.sample(frac = 1)
data = data[:5000]

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

# Vectorization -- TF-IDF
TFIDF_Vect = TfidfVectorizer(analyzer = clean_text)
X_TFIDF = TFIDF_Vect.fit_transform(data['review'])
X_TFIDF_pd = pd.DataFrame(X_TFIDF.toarray(), columns = TFIDF_Vect.get_feature_names())

warnings.filterwarnings('ignore', category = DeprecationWarning)

rf = RandomForestClassifier()
param = {'n_estimators' : [10, 150, 300],
         'max_depth' : [30, 60, 90, None]}

gs = GridSearchCV(rf, param, cv = 3, n_jobs = 4)
gs_fit = gs.fit(X_TFIDF_pd, data['sentiment'])
result1 = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending = False)

svc = SVC(kernel = 'rbf' , probability=True)
param = {'gamma' : [0.1, 0.5, 1],
         'C' : [1, 5, 10]}

gs2 = GridSearchCV(svc, param, cv = 3, n_jobs = 4)
gs_fit2 = gs2.fit(X_TFIDF_pd, data['sentiment'])
result2 = pd.DataFrame(gs_fit2.cv_results_).sort_values('mean_test_score', ascending = False)

gbc = GradientBoostingClassifier()
param = {'n_estimators' : [100, 300, 500]}

gs3 = GridSearchCV(gbc, param, cv = 3, n_jobs = 4)
gs_fit3 = gs3.fit(X_TFIDF_pd, data['sentiment'])
result3 = pd.DataFrame(gs_fit3.cv_results_).sort_values('mean_test_score', ascending = False)

rf_scores = result1['mean_test_score']
svm_scores = result2['mean_test_score']
gb_scores = result3['mean_test_score']

rf_scores = np.array(rf_scores)[:3]
svm_scores = np.array(svm_scores)[:3]
gb_scores = np.array(gb_scores)[:3]

print(rf_scores)
print(svm_scores)
print(gb_scores)

x = np.array([1,2,3])
plt.figure()
plt.scatter(x, rf_scores, c = 'r')
plt.scatter(x, svm_scores, c = 'b')
plt.scatter(x, gb_scores, c = 'g')
plt.xlabel("Order")
plt.ylabel("Score")
plt.show()