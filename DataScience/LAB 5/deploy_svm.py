import pandas as pd
import string
import re
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

import nltk
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'body_text'], header=None)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

def clean_text(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(' ')), 3) * 100

# calculate the length of messages excluding spaces
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_fit = tfidf_vect.fit(data['body_text'])
X_tfidf = tfidf_fit.transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf_feat, data['label'], test_size=0.2)

svc = SVC(kernel = 'rbf' ,gamma = 0.5, C = 5, probability=True)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('Prediction accuracy: ', accuracy_score(y_test, y_pred))

pickle.dump(tfidf_fit, open('svc_tfidf.pkl', 'wb'))
pickle.dump(svc , open('svc_trained.pkl', 'wb'))
