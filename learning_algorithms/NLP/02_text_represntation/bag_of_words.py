from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "cancer cell growth",
    "cancer cell mutation"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents) 
# fit =builds/ learn  the vocabulary 
# transform = convert the documnet into vetors 

print(vectorizer.get_feature_names_out())
print(X.toarray())