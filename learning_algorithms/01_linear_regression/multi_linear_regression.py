"""
in linear regression there is only one input feature and one output 

y^​=w1​x1​+w2​x2​+⋯+wp​xp​+b
many input featuers and one output
each feature gets it own weight 
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. CREATE DATA
X = np.array([
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 5],
    [1800, 3, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2500, 5, 1],
    [2800, 5, 1],
    [3000, 6, 1],
    [3500, 6, 1]
])

y = np.array([
    200,
    230,
    300,
    350,
    400,
    450,
    520,
    570,
    620,
    700
])

# 2. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. CREATE MODEL
model = LinearRegression()

# 4. TRAIN MODEL
model.fit(X_train, y_train)

# 5. LEARNED PARAMETERS
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# 6. MAKE PREDICTIONS
y_pred = model.predict(X_test)

print("Actual values:", y_test)
print("Predicted values:", y_pred)

# 7. CALCULATE MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)