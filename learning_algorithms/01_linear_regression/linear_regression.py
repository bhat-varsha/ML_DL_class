"""
Linear Regression is a supervised machine-learning algorithm used to predict a continuous numerical value
 by learning a relationship between input features and a numerical target.

 Whenever the target is a continuous numerical value and a linear relationship is a reasonable starting assumption.
    THE MODELS FITS STRIGHT LINE 

 try to find the relationship between x and y 

 y = w * x + b or y^​=b0​+b1​x
    w or b1 is the weight or coefficient of the feature x
    B0 or b is the bias or intercept term
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  #linear module contaien logistic regression , Ridge , Lasso 
from sklearn.metrics import mean_squared_error  #metrices to measure model performance 

# 1. CREATE DATA
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

# 2. VISUALIZE THE DATA TO CHECK IF MODELS FITS A STRAIGHT LINE 
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Input Data")
plt.show()

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # meaning 20% test and 80% train
    random_state=42
)

# 4. CREATE THE MODEL
model = LinearRegression()

# 5. TRAIN THE MODEL
model.fit(X_train, y_train)

# 6. LOOK AT LEARNED PARAMETERS
print("Weight:", model.coef_)
print("Bias:", model.intercept_)

# 7. MAKE PREDICTIONS
y_pred = model.predict(X_test)
print("Actual values:", y_test)
print("Predicted values:", y_pred)

# 8. CALCULATE MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 9. VISUALIZE THE REGRESSION LINE
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()