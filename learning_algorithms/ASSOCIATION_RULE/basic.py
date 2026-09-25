# pip install mlxtend

import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#created trnasaction data 
transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Eggs"],
    ["Milk", "Bread", "Butter"]
]
# the above is natural transaction representatiion , but the apriori need binary matrix
#so we use TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(
    te_array,
    columns=te.columns_
)
print(df)
# we will get one hot encoded transaction matrix ,
# One-Hot Encoding is a technique for representing categorical values as binary vectors, 


#run the apriori
# 
# frequent itemsets 
frequent_itemsets = apriori(
    df,
    min_support=0.4,
    use_colnames=True
)
print(frequent_itemsets)

#generate association rules 
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
print(rules)

#filtering the useful rules
rules = rules[
    (rules["confidence"] >= 0.7) &
    (rules["lift"] > 1)
]

#final output 
print("\nFiltered Rules:")
print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]
)