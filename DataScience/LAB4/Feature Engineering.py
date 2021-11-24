import string
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_colwidth', 100)  # Extend column width

stopwords = nltk.corpus.stopwords.words('english')  # remove Stop words
ps = nltk.PorterStemmer()  # make stem words

data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'body_text'], header=None)

def clean_text(text):
    # Remove Punctuation
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    # make Token
    tokens = re.split('\W+', text)
    # Remove StopWords and make Stem
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

data['body_len'] = data['body_text'].apply(lambda x : len(x) - x.count(' '))

def count_punct(text):
    count = sum( [1 for char in text if char in string.punctuation ])
    return round(count / (len(text) - text.count(' ')) , 3)* 100

data['punc%'] = data['body_text'].apply(lambda x : count_punct(x))

bins = np.linspace(0, 200, 40)

plt.hist(data[data['label'] == 'spam']['body_len'], bins, alpha = 0.5 , density = True, stacked = True,label = 'spam')
plt.hist(data[data['label'] == 'ham']['body_len'], bins, alpha = 0.5, density = True, stacked = True, label = 'ham')
plt.legend(loc = 'upper left')
plt.show()

bins = np.linspace(0, 50, 40)

plt.hist(data[data['label'] == 'spam']['punc%'], bins, alpha = 0.5 , density = True, stacked = True,label = 'spam')
plt.hist(data[data['label'] == 'ham']['punc%'], bins, alpha = 0.5, density = True, stacked = True, label = 'ham')
plt.legend(loc = 'upper left')
plt.show()