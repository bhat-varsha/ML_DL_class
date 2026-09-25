from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

data = {
    'Iron Man': [1, 0, 1, 1, 0],
    'Captain America': [1, 1, 0, 1, 0],
    'Thor': [0, 1, 1, 1, 0],
    'Hulk': [0, 0, 1, 1, 1],
    'Black Widow': [1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)
df = pd.read_csv("counts.csv")

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""
for the same code , use the counts.csv file import that ,and convert that to dataframe 
then do the Apriori to check the frequent genes present in the counts.csv


"""