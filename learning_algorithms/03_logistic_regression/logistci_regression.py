"""
linear = predicts continous nunber
logistci =predicts class or probability

Logistic Regression is a supervised machine learning classification algorithm used to predict the probability of an outcome belonging to a class.
mainly for classification prblms 

here in the logistci function , we will use sigmoud
for logistic we have to get output in 0 or 1 output 
if we get 0.10 like that here we will use sigmodin fucntion 

steps of logistic :
1. first caclualte linear equation (weight and bias)
2. then that score pass through sigmodi (which converts any value of linear score between 0 and 1 )
3. then the threshold we set for the sigmodin value

Linear combination + sigmoid + classification threshold

DECISION BOUNDRY : APPEARNS IN CLASSIFICATION MODELS WHEN WE WANT THE OUTPUT IN CLASSES OR CATEGRORY FORM
IT IS CLASSIFICATION THRESHOLD

baisc flow of logistic
Input
z = wx + b(linear score)
Sigmoid
Probability
Threshold
predcit Class

but here is also we have to do loss fucniton 
here we use Binary Cross Entropy / log loss
Binary Cross-Entropy (BCE) is a loss function used in binary classification to measure how far the model's predicted probabilities are from the actual labels.
L=-[ylog(p)+(1-y)log(1-p)]

then after loss fucntion comes gradient descent 
Gradient tells us the direction in which the loss increases most rapidly.
Gradient Descent is an optimization algorithm used to minimize the loss function by repeatedly adjusting the model's parameters in the direction that reduces the loss.

regularizition : prevents the model making weigths unnecessarily large, helpogin control overfitting

lasso  :psuh the weights to zero
ridge : shrinks the weight to zero

in sklearn , while explaining the model , weill use this 
penalty="l2"
C=1.0 : which is iverse regularization strenght


LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000
)
penalty → type of regularization
C → regularization strength control
solver → optimization algorithm
max_iter → maximum optimization iterations

"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#Logistic Regression is trained through an iterative optimization algorithm

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability prediction
y_prob = model.predict_proba(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))