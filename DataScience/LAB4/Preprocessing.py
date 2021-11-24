import re
import nltk
import string
import pandas as pd

# .tsv file --- csv와 비슷하지만 comma 대신 tab으로 구분된 file
data = pd.read_csv('data/SMSSpamCollection.tsv', sep='\t', names= ['label', 'body_text'], header = None)

def remove_punct(text):
    text_nopunct = ''.join([char for char in text if char not in string.punctuation])
    return text_nopunct

def tokenize(text):
    tokens = re.split('\W+', text) # W+ means either a word character or a dash can go there
    return tokens

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords]
    return text

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

def lemmatising(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_clean'] = data['body_text'].apply(lambda x : remove_punct(x))
data['body_text_tokenized']  = data['body_text_clean'].apply(lambda x : tokenize(x.lower()))

# Remove Stop Words --- in other words ... remove common words like a, an, the, is , ..
stopwords = nltk.corpus.stopwords.words('english')
data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x : remove_stopwords(x))

# Stemming data --- origin form of words like having -> have , flying -> fly
ps = nltk.PorterStemmer()
data['text_body_stemmed'] = data['body_text_nostop'].apply(lambda x : stemming(x))

# Lemmatising --- Stemming 보다 느리지만 더 정확하게 어원으로
wn = nltk.WordNetLemmatizer()
data['body_text_lemmatised'] = data['body_text_nostop'].apply(lambda x : lemmatising(x))






