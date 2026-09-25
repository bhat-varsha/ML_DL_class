from collections import Counter  #to count word frequency

import sklearn
import matplotlib as plt
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

#open the file:
with open('nlp.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#preprocessing :
#making all the leterr lower case
text=text.lower()
print(text)

#tokenizartion , making each word as single entity 
tokens = word_tokenize(text) 
print(tokens)
print("Number of tokens:", len(tokens))

#clean the tokens (stopwrods ,spaces)

#remove the punctuation :
tokens_clean = []
for token in tokens:
    if token not in string.punctuation:
        tokens_clean.append(token)
print(tokens_clean)

#remove the stopwrods
stop_words = set(stopwords.words('english'))
filtered_tokens = []
for token in tokens_clean:
    if token not in stop_words:
        filtered_tokens.append(token)
print(filtered_tokens)

#to calculate the word frequency:
word_frequency = Counter(filtered_tokens)
print(word_frequency)
print(word_frequency.most_common(20))

#fro visualization purpose : 
common_words = word_frequency.most_common(10)
words = [item[0] for item in common_words]
counts = [item[1] for item in common_words]
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Common Words')
plt.xticks(rotation=45)
plt.show()



