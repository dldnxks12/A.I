import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore', category = DeprecationWarning)

# TF-IDF Data
tfidf_vect = TfidfVectorizer(analyzer = clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_len'] , data['punc%'], pd.DataFrame(X_tfidf.toarray())], axis = 1)

# BOG
count_vect = CountVectorizer(analyzer = clean_text)
X_count = count_vect.fit_transform(data['body_text'])
X_count_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(X_count.toarray())], axis = 1)

rf = RandomForestClassifier()

param = {'n_estimators' : [10, 150, 300],
         'max_depth' : [30, 60, 90, None]}

gs = GridSearchCV(rf, param, cv = 5, n_jobs = 4)
gs_fit = gs.fit(X_count_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending = False).head()

rf2 = RandomForestClassifier()
gs = GridSearchCV(rf2, param, cv = 5, n_jobs = 4)
gs_fit = gs.fit(X_tfidf_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending = False).head()