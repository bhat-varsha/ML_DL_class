# Association Rule  Mining for gene-expression patterns;

from Bio import Entrez
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
Entrez.email = "bhatvarsh08@gmail.com"

handle = Entrez.esearch(db="pubmed",term = "BRCA1" , retmax=1)
record = Entrez.read(handle)
handle.close()

handle = Entrez.efetch(db="pubmed",id=record["IdList"][0],rettype="abstract",retmode="text")

data = handle.read().split("\n\n")
handle.close()

print(data)

stopwords = {
    "the", "a", "an", "and", "or", "but",
    "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "as",
    "that", "this", "these", "those",
    "it", "its", "they", "them", "their",
    "he", "she", "his", "her",
    "we", "our", "you", "your",
    "i", "me", "my",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "may", "might",
    "will", "would", "should",
    "not", "no",
    "than", "then", "also",
    "which", "who", "whom",
    "what", "when", "where", "why", "how"
}
transactions = []
for text in data:
    words = text.lower().split()
    words = [w.strip(".,()") for w in words]        #remove punctuation
    words = [w for w in words if w not in stopwords and len(w) > 3]
    transactions.append(list(set(words)))            #unique words only

print(transactions)

encoder = TransactionEncoder() 
encoded_array = encoder.fit(transactions).transform(transactions)
print(encoded_array)
df = pd.DataFrame(encoded_array, columns=encoder.columns_)

df = pd.DataFrame(encoded_array, columns=encoder.columns_).astype(int)

#now we utilze apriori algorithm to find frequent itemsets
from mlxtend.frequent_patterns import apriori , association_rules 

frequent_words = apriori(df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_words, metric="confidence", min_threshold=0.7)

print(rules[['antecedents', 'consequents',]])