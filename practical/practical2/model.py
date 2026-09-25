#random forest and svm model 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# SVM
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,  f1_score,
    roc_auc_score, classification_report, roc_curve  ,  confusion_matrix
)
from sklearn.model_selection import GridSearchCV

#improting the file

# Import VCF
vcf_file = "demo_snp_gwas_rf_svm.vcf"

#preprocessing of vcf file :
with open(vcf_file, "r") as f:
    for line in f:
        if line.startswith("#CHROM"):
            header = line.rstrip().split("\t")
            break

print(header)

header_line = None
with open(vcf_file, "r") as f:
    for i, line in enumerate(f):
        if line.startswith("#CHROM"):
            header_line = i
            break
        if header_line is not None:
            raise ValueError("vcf file not contain header line startign with #chrom")


vcf = pd.read_csv(vcf_file, sep="\t", skiprows=header_line, dtype=str)
vcf = vcf.rename(columns={
    "#CHROM": "CHROM",
    "POS": "POS",
    "ID": "ID",
    "REF": "REF",
    "ALT": "ALT",
    "QUAL": "QUAL",
    "FILTER": "FILTER",
    "INFO": "INFO",
    "FORMAT": "FORMAT"
})

print("VCF shape:", vcf.shape)
print(vcf.head())
display(vcf.head())

fixed_columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL","FILTER","INFO","FORMAT"]

sample_columns = [c for c in vcf.columns if c not in fixed_columns]
print(sample_columns)

def genotype_to_numeric(value):
    if pd.isna(value):
        return np.nan

    gt = str(value).split(":")[0]

    mapping = {
        "0/0": 0,
        "0|0": 0,
        "0/1": 1,
        "1/0": 1,
        "0|1": 1,
        "1|0": 1,
        "1/1": 2,
        "1|1": 2
    }

    return mapping.get(gt, np.nan)

genotype = pd.DataFrame(index=vcf.index)

for sample in sample_columns:
    genotype[sample] = vcf[sample].apply(genotype_to_numeric)
display(genotype.head())


#big part

# convertind to string type variable , snp data
#attaching label
vcf["SNP_ID"] = (
    vcf["CHROM"].astype(str) + ":" +
    vcf["POS"].astype(str) + ":" +
    vcf["REF"].astype(str) + ":" +
    vcf["ALT"].astype(str)
)
genotype.index = vcf["SNP_ID"]
display(genotype.head())

#transposing
X_genotype = genotype.T.copy()
display(X_genotype.head())

#missing values
missing_rate = X_genotype.isna().mean()
X_qc = X_genotype.loc[:, missing_rate <= 0.20].copy()

#qualtiy check of snps
#Removing all snps without any hits
variable_snps = X_qc.nunique(dropna=True) > 1
X_qc = X_qc.loc[:, variable_snps]

X_qc = X_qc.fillna(X_qc.median()) #handling the missing values

print(X_genotype.shape[1])
print(X_qc.shape[1])
display(X_qc)


#to import csv file
PHENOTYPE_FILE="demo_phenotype.csv"
phenotype = pd.read_csv(PHENOTYPE_FILE)

display(phenotype)
print("Phenotype data:")
print(phenotype.head())


phenotype = phenotype.set_index("Sample")

common_samples = X_qc.index.intersection(phenotype.index)

X = X_qc.loc[common_samples]
y = phenotype.loc[common_samples, "Phenotype"]

print("Number of common samples:", len(common_samples))
print("Common samples:", common_samples.tolist())

print("X shape:", X.shape)
print("y shape:", y.shape)

label_map = {"Control": 0, "Disease": 1}
y_encoded = y.map(label_map)

if y_encoded.isna().any():
    raise ValueError("Went wrong")

print(y_encoded)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.33,
    random_state=42,
    stratify=y_encoded
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\nTraining class counts:")
print(y_train.value_counts())
print("\nTesting class counts:")
print(y_test.value_counts())

# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Predictions:")
print(y_pred_rf)

y_prob_rf = rf_model.predict_proba(X_test)
print("\nRandom Forest Probabilities:")
print(y_prob_rf)


# Probability of Disease class (class = 1)
y_prob_rf_disease = rf_model.predict_proba(X_test)[:, 1]

print("\nDisease probabilities:")
print(y_prob_rf_disease)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)


print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["Control", "Disease"]
))
print(classification_report(y_test, y_pred_rf, target_names=["Control", "Disease"]))

# feautre importacnes
#which feature was more importent 

rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
display(rf_importance)
print(rf_importance)


# SVM
#utilizign the pipeline , we can do two diffrent algoirthms 
#pipeline of algorithsms first scaler , and then svm 

#svm , to wokr on hyperplanes
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf", # dimension , descision boundry between classes , rbf is kernel trick algortihms
        #intorduction of higher dimenesions 
        probability=True,
        random_state=42
    ))
])
# Train
svm_model.fit(X_train, y_train)
# Predict
y_pred_svm = svm_model.predict(X_test)

print("SVM Predictions:")
print(y_pred_svm)

y_prob_svm = svm_model.predict_proba(X_test)

print("\nSVM Probabilities:")
print(y_prob_svm)


# Probability of Disease class (class = 1)
y_prob_svm_disease = svm_model.predict_proba(X_test)[:, 1]

print("\nDisease probabilities:")
print(y_prob_svm_disease)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:")
print(svm_accuracy)

print("\nSVM Classification Report:")

print(classification_report(
    y_test,
    y_pred_svm,
    target_names=["Control", "Disease"]
))


#confusin matrix we have to build 
# RANDOM FOREST CONFUSION MATRIX
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(cm_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=["Control", "Disease"],
    yticklabels=["Control", "Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# SVM CONFUSION MATRIX
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Confusion Matrix:")
print(cm_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    xticklabels=["Control", "Disease"],
    yticklabels=["Control", "Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# grod search cv 
#hyperparameter tuning for random forest

rf_grid = {
    "n_estimators": [100, 200], "max_depth": [None, 3, 5], "min_samples_split": [2, 4]
}
rf_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"), rf_grid, cv=3, scoring="f1",n_jobs=-1)

rf_search.fit(X_train,y_train)

print("Best SVM parameters:")
print(rf_search.best_params_)

#svm hyperpapramets
svm_grid = {
    "svm__kernel": ["rbf", "linear"]
}
svm_search = GridSearchCV(svm_model,svm_grid,cv=3,scoring="f1",n_jobs=-1)
svm_search.fit(X_train,y_train)

print("Best SVM parameters:")
print(svm_search.best_params_)

#############################################
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }
best_rf = rf_search.best_estimator_
best_svm = svm_search.best_estimator_

best_rf_pred = best_rf.predict(X_test)
best_rf_prob = best_rf.predict_proba(X_test)[:, 1]

best_svm_pred = best_svm.predict(X_test)
best_svm_prob = best_svm.predict_proba(X_test)[:, 1]

tuned_results = pd.DataFrame([
    evaluate_model("Tuned Random Forest", y_test, best_rf_pred, best_rf_prob),
    evaluate_model("Tuned SVM", y_test, best_svm_pred, best_svm_prob)
])

display(tuned_results.round(3))

###############     HAVE TO BUILD ROC CURVE
# ROC CURVE
# Random Forest ROC
fpr_rf, tpr_rf, thresholds_rf = roc_curve(
    y_test,
    best_rf_prob
)
rf_auc = roc_auc_score(
    y_test,
    best_rf_prob
)


# SVM ROC
fpr_svm, tpr_svm, thresholds_svm = roc_curve(
    y_test,
    best_svm_prob
)
svm_auc = roc_auc_score(
    y_test,
    best_svm_prob
)

# PLOT ROC CURVES
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)
plt.plot(
    fpr_svm,
    tpr_svm,
    label=f"SVM (AUC = {svm_auc:.3f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest vs SVM")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()