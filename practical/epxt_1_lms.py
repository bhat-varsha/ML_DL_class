# IMPORT LIBRARIES
import GEOparse  
#GEOparse is a Python library used to access and work with GEO (Gene Expression Omnibus) datasets.
import pandas as pd #used for handling tabular data.(excel spreadhseet form)
#help to create and manipulate this type of table
import numpy as np  #used for numerical calculations and arrays.
import matplotlib.pyplot as plt  #used to create graphs and visualizations.

# scikit-learn Python machine-learning library.
#scikit learn libraries for preprocessing, feature selection, and dimensionality reduction
from sklearn.preprocessing import StandardScaler  #for strandization or sclaing of features.
#for each features , it transforms the values approximately to have a mean of 0 and a standard deviation of 1.
from sklearn.feature_selection import VarianceThreshold  #for feature selection based on variance.
from sklearn.feature_selection import SelectKBest, f_classif 
 #selectKBest is used to select the top k features based on a scoring function. 
 #f_classif is a scoring function that computes the ANOVA F-value for each feature.
from sklearn.decomposition import PCA   #pca is used to reduce the dimensionality of the dataset while preserving as much variance as possible.

# a. DOWNLOAD AND EXPLORE DATASET
gse = GEOparse.get_GEO(geo="GSE58911",destdir=".")
print("Number of samples:", len(gse.gsms))
print("Number of platforms:", len(gse.gpls))
sample_id = list(gse.gsms.keys())[0]
print("\nSample ID:", sample_id)
print("\nSample metadata:")
print(gse.gsms[sample_id].metadata)

# b. EXTRACT SAMPLE METADATA AND IDENTIFY GROUPS

metadata_list = []
for sample_id, sample in gse.gsms.items():
    metadata = sample.metadata
    metadata_list.append({
        "Sample_ID": sample_id,
        "Title": metadata.get("title", [""])[0],
        "Source": metadata.get("source_name_ch1", [""])[0]
    })
metadata_df = pd.DataFrame(metadata_list)
print("\nSample metadata:")
print(metadata_df.head())

# Identify biological groups
def identify_group(title, source):
    text = (
        str(title) + " " +
        str(source)
    ).lower()
    if "hnscc" in text:
        return "Tumor"
    elif "normal" in text :
        return "Normal"
    else:
        return "Unknown"

metadata_df["Group"] = metadata_df.apply(
    lambda row: identify_group(
        row["Title"],
        row["Source"]
    ),
    axis=1
)
print("\nBiological groups:")
print(metadata_df["Group"].value_counts())


# Remove unknown samples
metadata_df = metadata_df[
    metadata_df["Group"].isin(
        ["Tumor", "Normal"]
    )
].copy()

# c. CONSTRUCT EXPRESSION MATRIX
expression = gse.pivot_samples("VALUE")
print("\nOriginal expression shape:")
print(expression.shape)

# Samples as rows
# Features as columns
X = expression.T
print("\nExpression matrix shape:")
print(X.shape)
print("\nExpression matrix:")
print(X.head())

# Match expression samples with metadata
common_samples = X.index.intersection(
    metadata_df["Sample_ID"]
)
X = X.loc[common_samples]
metadata_df = metadata_df[
    metadata_df["Sample_ID"].isin(common_samples)
].copy()
metadata_df = metadata_df.set_index(
    "Sample_ID"
)
metadata_df = metadata_df.loc[X.index]
print("\nFinal expression shape:")
print(X.shape)

# d. HANDLE MISSING VALUES AND DUPLICATE FEATURES
# Convert to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)
# Check missing values
print("\nTotal missing values:")
print(X.isnull().sum().sum())

# Remove features having more than 10% missing values
X = X.dropna(
    axis=1,
    thresh=int(0.90 * len(X))
)
# Fill remaining missing values with median
X = X.fillna(
    X.median()
)
print("\nMissing values after handling:")
print(X.isnull().sum().sum())

# Check duplicate feature names
print("\nDuplicate feature names:")
print(X.columns.duplicated().sum())

# Remove duplicate feature names
X = X.loc[
    :,
    ~X.columns.duplicated()
]
# Remove duplicate feature profiles
X = X.loc[
    :,
    ~X.T.duplicated()
]
print("\nShape after duplicate removal:")
print(X.shape)

# e. EXAMINE DISTRIBUTION AND TRANSFORMATION
plt.figure(figsize=(8, 5))
plt.hist(
    X.iloc[:, 0],
    bins=50
)
plt.xlabel("Expression")
plt.ylabel("Frequency")
plt.title("Expression Distribution")
plt.show()

# GSE58911 expression values are already
# RMA-normalized and log2 transformed.
# Therefore, no additional log transformation is performed.

# f. NORMALIZATION / SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(
    X_scaled,
    index=X.index,
    columns=X.columns
)
print("\nScaled expression matrix:")
print(X_scaled.head())

# g. FEATURE SELECTION
# -------- METHOD 1: VARIANCE --------
feature_variance = X_scaled.var(axis=0)
top_features = (
    feature_variance
    .sort_values(ascending=False)
    .head(5000)
    .index
)
X_variance = X_scaled.loc[:, top_features]
print("\nVariance-selected features:")
print(X_variance.shape)
# -------- METHOD 2: ANOVA F-TEST --------
y = metadata_df["Group"].map({
    "Normal": 0,
    "Tumor": 1
})
selector = SelectKBest(
    score_func=f_classif,
    k=5000
)
X_anova_array = selector.fit_transform(X_scaled,y)
anova_features = X_scaled.columns[selector.get_support()]
X_anova = pd.DataFrame(
    X_anova_array,
    index=X_scaled.index,
    columns=anova_features
)
print("\nANOVA-selected features:")
print(X_anova.shape)

# h. PCA AND VISUALIZATION
# -------- PCA USING VARIANCE FEATURES --------
pca_variance = PCA(n_components=2)
pca_variance_result = pca_variance.fit_transform(
    X_variance
)
pca_variance_df = pd.DataFrame(
    pca_variance_result,
    index=X_variance.index,
    columns=["PC1", "PC2"]
)
pca_variance_df["Group"] = metadata_df["Group"]
plt.figure(figsize=(8, 5))

for group in ["Normal", "Tumor"]:
    data = pca_variance_df[
        pca_variance_df["Group"] == group
    ]
    plt.scatter(
        data["PC1"],
        data["PC2"],
        label=group
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Variance Selected Features")
plt.legend()
plt.show()
# -------- PCA USING ANOVA FEATURES --------
pca_anova = PCA(
    n_components=2
)
pca_anova_result = pca_anova.fit_transform(
    X_anova
)
pca_anova_df = pd.DataFrame(
    pca_anova_result,
    index=X_anova.index,
    columns=["PC1", "PC2"]
)
pca_anova_df["Group"] = metadata_df["Group"]
plt.figure(figsize=(8, 5))
for group in ["Normal", "Tumor"]:
    data = pca_anova_df[
        pca_anova_df["Group"] == group
    ]
    plt.scatter(
        data["PC1"],
        data["PC2"],
        label=group
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - ANOVA Selected Features")
plt.legend()
plt.show()

# i. COMPARE FEATURE SELECTION APPROACHES
print("\nPCA variance explained - Variance method:")
print(
    pca_variance.explained_variance_ratio_
)
print("\nPCA variance explained - ANOVA method:")
print(
    pca_anova.explained_variance_ratio_
)
comparison = pd.DataFrame({

    "Method": [
        "Variance",
        "ANOVA"
    ],

    "Number_of_Features": [
        X_variance.shape[1],
        X_anova.shape[1]
    ],

    "PC1_Variance": [
        pca_variance.explained_variance_ratio_[0],
        pca_anova.explained_variance_ratio_[0]
    ],

    "PC2_Variance": [
        pca_variance.explained_variance_ratio_[1],
        pca_anova.explained_variance_ratio_[1]
    ]

})
print("\nComparison:")
print(comparison)

# j. CREATE AND EXPORT ML-READY DATASET
# Variance-selected ML dataset
ml_variance = X_variance.copy()
ml_variance["Target"] = y

# ANOVA-selected ML dataset
ml_anova = X_anova.copy()
ml_anova["Target"] = y

# Export
ml_variance.to_csv(
    "GSE58911_ML_variance.csv"
)
ml_anova.to_csv(
    "GSE58911_ML_ANOVA.csv"
)
metadata_df.to_csv(
    "GSE58911_metadata.csv"
)
print("\nML-ready datasets exported successfully.")

# k. PREPROCESSING / FEATURE ENGINEERING SUMMARY
print ("""
Preprocessing justification:

1. Missing values were checked and features with more than
   10% missing values were removed.

2. Remaining missing values were replaced using the median.

3. Duplicate feature names and duplicate expression profiles
   were removed.

4. The dataset is already RMA-normalized and log2 transformed,
   so an additional log transformation was not performed.

5. StandardScaler was used to standardize features before PCA
   and machine-learning analysis.

6. Two feature-selection methods were used:
   - Variance-based selection
   - ANOVA F-test

7. PCA was performed separately for both feature-selection
   approaches to compare the biological group separation.

8. The resulting feature matrices with biological group labels
   were exported as ML-ready datasets.
""")