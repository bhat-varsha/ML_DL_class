import GEOparse
gse = GEOparse.get_GEO(
    geo="GSE10072",
    destdir="."
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print ("length of sample",len(gse.gsms))
print("Number of platforms:", len(gse.gpls)) #This gave me platform data.
#To get 1 sample ID and its metadata, you can use the following code:
sample_id = list(gse.gsms.keys())[0]
print("my sample ids are:", sample_id)
print(gse.gsms[sample_id].metadata)

expression = gse.pivot_samples('VALUE')  #createde data table 
print("Expression shape:", expression.shape)
print("Expression data:")
print(expression.head())

sample = gse.gsms["GSM254625"]
print(expression.info())
print(sample.table.head())
print(sample.table.shape)

#transpose it
X = expression.T
print(X.shape)
print(X.head())

#preprocessing 
#check missing values
missing = X.isnull().sum()

print(missing)

#drop na
X = X.dropna(axis=1, thresh=int(0.9 * len(X)))

print(X.shape)

X = X.fillna(X.median())


plt.figure(figsize=(8,5))
plt.hist(X.iloc[:, 0], bins=50)
plt.xlabel("Expression")
plt.ylabel("Frequency")
plt.title("Expression Before Log Transformation")
plt.show()
# ==========================================
# STEP 3: LOG TRANSFORMATION
# ==========================================
X_log = np.log2(X + 1)

print("\nAfter Log2 Transformation:")
print(X_log.head())

plt.figure(figsize=(8,5))
plt.hist(X_log.iloc[:, 0], bins=50)
plt.xlabel("log2(Expression + 1)")
plt.ylabel("Frequency")
plt.title("Expression After Log Transformation")
plt.show()

feature_variance = X_log.var(axis=0)

print(feature_variance.describe())