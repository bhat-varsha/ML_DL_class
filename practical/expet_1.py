import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# STEP 1: LOAD GEO DATA
# ==========================================

gse = GEOparse.get_GEO(
    geo="GSE10072",
    destdir="."
)

print("Length of samples:", len(gse.gsms))
print("Number of platforms:", len(gse.gpls))


# ==========================================
# STEP 2: EXPLORE SAMPLE DATA
# ==========================================

sample_id = list(gse.gsms.keys())[0]

print("\nSample ID:", sample_id)
print("Sample metadata:")
print(gse.gsms[sample_id].metadata)


# ==========================================
# STEP 3: CREATE EXPRESSION MATRIX
# ==========================================

expression = gse.pivot_samples('VALUE')

print("\nExpression shape:", expression.shape)

print("\nExpression data:")
print(expression.head())

print("\nExpression information:")
print(expression.info())


# ==========================================
# STEP 4: TRANSPOSE EXPRESSION MATRIX
# ==========================================

X = expression.T

print("\nTransposed expression shape:", X.shape)
print("\nTransposed expression data:")
print(X.head())


# ==========================================
# STEP 5: CHECK MISSING VALUES
# ==========================================

missing = X.isnull().sum()

print("\nMissing values:")
print(missing)


# ==========================================
# STEP 6: REMOVE FEATURES WITH MANY
#         MISSING VALUES
# ==========================================

X = X.dropna(
    axis=1,
    thresh=int(0.9 * len(X))
)

print("\nShape after removing features with >10% missing values:")
print(X.shape)


# ==========================================
# STEP 7: FILL REMAINING MISSING VALUES
#         USING MEDIAN
# ==========================================

X = X.fillna(X.median())

print("\nMissing values after median imputation:")
print(X.isnull().sum().sum())


# ==========================================
# STEP 8: BEFORE LOG TRANSFORMATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    X.iloc[:, 0],
    bins=50
)

plt.xlabel("Expression")
plt.ylabel("Frequency")
plt.title("Expression Before Log Transformation")

plt.show()
plt.close()

# ==========================================
# STEP 9: LOG2 TRANSFORMATION
# ==========================================

X_log = np.log2(X + 1)

print("\nExpression data after Log2 transformation:")
print(X_log.head())


# ==========================================
# STEP 10: AFTER LOG TRANSFORMATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    X_log.iloc[:, 0],
    bins=50
)

plt.xlabel("log2(Expression + 1)")
plt.ylabel("Frequency")
plt.title("Expression After Log Transformation")

plt.show()
plt.close()


# ==========================================
# STEP 11: FEATURE VARIANCE
# ==========================================

feature_variance = X_log.var(axis=0)

print("\nFeature variance summary:")
print(feature_variance.describe())

top_features = feature_variance.sort_values(ascending=False).head(5000).index
X_hvg = X_log[top_features]
#=================================
#STRANDADIZATION
#=========================s
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_hvg)

X_scaled = pd.DataFrame(
    X_scaled,
    index=X_hvg.index,
    columns=X_hvg.columns
)
print(X_scaled.head())

print("\nMean of scaled features:")
print(X_scaled.mean())

print("\nStandard deviation of scaled features:")
print(X_scaled.std())