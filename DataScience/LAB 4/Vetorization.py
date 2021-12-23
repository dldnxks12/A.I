import pandas as pd
import re
import string
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_colwidth', 100)  # Extend column width

stopwords = nltk.corpus.stopwords.words('english')  # remove Stop words
ps = nltk.PorterStemmer()  # make stem words

data = pd.read_csv('../LAB 5/SMSSpamCollection.tsv', sep='\t', names=['label', 'body_text'], header=None)

def clean_text(text):
    # Remove Punctuation
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    # make Token
    tokens = re.split('\W+', text)
    # Remove StopWords and make Stem
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

# BOG
count_vect = CountVectorizer(analyzer  = clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
X_counts_df = pd.DataFrame(X_counts.toarray(), columns = count_vect.get_feature_names())

# N-Grams
ngram_vect = CountVectorizer(ngram_range = (2,2), analyzer = clean_text)
X_counts2 = ngram_vect.fit_transform(data['body_text'])
X_counts_df = pd.DataFrame(X_counts2.toarray(), columns = ngram_vect.get_feature_names())

# TF IDF
tfidf_vect = TfidfVectorizer(analyzer = clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_pd = pd.DataFrame(X_tfidf.toarray(), columns = tfidf_vect.get_feature_names())

