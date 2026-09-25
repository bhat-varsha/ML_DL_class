"""
KNN :supervised learnign for both both rgeression  or classification

PREDCITED OUTPUT IS BASED ON NEARBY DATA POINTS

K = how many neighbors we look at(how many closest traningh samples )
Nearest = closest according to some distance measure
Neighbors = existing training data points

knn meausre nearest using euclidean distance 
d=(x1-y1)2+(x2-y2)2

workflow:
New point
Calculate distance to training points
Find K closest points
Majority vote
Prediction

Training
   ↓
KNN stores training data
   ↓
New/test point arrives
   ↓
Calculate distance to training points
   ↓
Select K nearest points
   ↓
Classification → majority class
Regression → average target value


decinding k is impot , if k is small , output wil be sensitgive
if k is big , underfit or give genralized answer

in knn scaler is imp(StandardScaler)

knn is a lazy learner :: during traing in all other algorithms , 
it will udnerstadn mathemtical fucntion and calculate weigth and bias
but in knn , it just asotres the trainig data

major computation happpen when we do predcit 
then calculate the distanc eusing eucledian , find neighbours vite the majority , and then precit

KNN Hyperparameters
KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2
)
n_neighbors → K
weights="uniform" → every neighbor has equal importance
weights="distance" → closer neighbors have more influence
metric → distance metric
p=2 → Euclidean distance
p=1 → Manhattan distance


"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()

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

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN model
model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
