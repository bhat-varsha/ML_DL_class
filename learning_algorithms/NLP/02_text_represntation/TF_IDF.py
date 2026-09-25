#term frequnecy 
#inverse documnet frequency 

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "cancer cell growth",
    "cancer cell mutation",
    "mutation causes abnormal growth"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
print(vectorizer.vocabulary_)