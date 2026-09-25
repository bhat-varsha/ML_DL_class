#TF  : TERM FREQUENCY : CHECK THOW FREQUENT A WORD IS  
#IDF : INVERSE DOCUMNET FREQUNECY :  REDUCES THE WEIGTAGE FOR FORMALLY REPEATIGN WORDS

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    'gene expression analysis in cancer',
    'cancer gene mutation analysis',
    'protein expression in cancer cells',
    'machine learning for gene expression'
]

print(documents)

#start assigning weight to particular words
#weigth assigned based in number of repetation 

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print("Shape of TF-IDF matrix:", X.shape)

print("Features:")
print(vectorizer.get_feature_names_out())
print("\nTF-IDF matrix:")
print(X.toarray())


#csv file , one column labels , another column text(about cancer)