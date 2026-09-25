import nltk  #natutal language toolkit
import string  # to remove the punctuation

from nltk.tokenize import word_tokenize  #can use sent_tokenize too
from nltk.corpus import stopwords  

#for stemming and lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

text = """
The patients were diagnosed with malignant tumors.
The tumors were treated successfully.
"""

# Lowercase
text = text.lower()

# Tokenization
tokens = word_tokenize(text)

# Remove punctuation
tokens = [
    word for word in tokens
    if word not in string.punctuation
]

# Stopword removal
stop_words = set(stopwords.words("english"))

tokens = [
    word for word in tokens
    if word not in stop_words
]

# Lemmatization
lemmatizer = WordNetLemmatizer()

tokens = [
    lemmatizer.lemmatize(word)
    for word in tokens
]

#stemming
stemmer = PorterStemmer()
words = ["playing", "played", "plays", "studies"]
for word in words:
    print(word, "→", stemmer.stem(word))


print(tokens)