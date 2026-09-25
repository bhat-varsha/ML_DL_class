"""
A straight-line model cannot properly capture the x^2 relationship.
Polynomial Regression solves this by creating polynomial features
  
y=wx+b      linear
y^​=w1​x1​+w2​x2​+w3​x3​+b  multiple 

y^​=w1​x+w2​x2+w3​x3+b   here is use X square and cube , that why polynomial 
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. CREATE NONLINEAR DATA
X = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10
]).reshape(-1, 1)

y = np.array([
    2, 5, 10, 17, 26,
    37, 50, 65, 82, 101
])


# 2. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. CREATE POLYNOMIAL TRANSFORMER
poly = PolynomialFeatures(degree=2)

# 4. TRANSFORM THE DATA
X_train_poly = poly.fit_transform(X_train)  
#fit → learn parameters/information from data
#transform → convert data into another representation

#x test is not gone fit , emans it will lead to data leakage , so we will only transform it
X_test_poly = poly.transform(X_test)

# 5. CREATE THE MODEL
model = LinearRegression()

# 6. TRAIN THE MODEL
model.fit(X_train_poly, y_train)

# 7. LOOK AT LEARNED PARAMETERS
print("Weights:", model.coef_)
print("Bias:", model.intercept_)


# 8. MAKE PREDICTIONS
y_pred = model.predict(X_test_poly)
print("Actual values:", y_test)
print("Predicted values:", y_pred)


# 9. EVALUATE THE MODEL
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)


# 10. VISUALIZE THE REGRESSION CURVE
X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)

y_curve = model.predict(X_curve_poly)

plt.scatter(X, y)
plt.plot(X_curve, y_curve)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression")
plt.show()