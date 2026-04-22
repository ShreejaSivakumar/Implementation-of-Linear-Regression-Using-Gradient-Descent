# Implementation-of-Linear-Regression-Using-Gradient-Descent
# DATE : 22/04/26

# NAME : SHREEJA R S 
# REF.NO : 25017561

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Gradient Descent (Linear Regression)

1. Load data: Read given CSV file (dataset) , take X = R&D Spend and Y = Profit  
2. Normalize: Scale X using mean and standard deviation by np.mean(X) , np.std(X)
3. Initialize: Set m = 0, b = 0, choose learning_rate and epochs  
4. Predict: Compute y_pred = m*X + b  
5. Update: Calculate gradients dm, db and adjust m, b  
6. Output and Plot: Print slope and intercept, plot regression line with data  


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:  SHREEJA R S 
RegisterNumber:  25017561
*/
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (read dataset)
data = pd.read_csv("50_Startups.csv")

# Select one feature (R & D Spend) and target value (Profit)
X = data['R&D Spend'].values
Y = data['Profit'].values

# Normalise (important for gradient descent)
X = (X - np.mean(X)) / np.std(X)

# Initialise the parameters 
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m*X + b
    
    # Gradient
    dm = (-2/n) * np.sum(X*(Y - y_pred))
    db = (-2/n) * np.sum(Y - y_pred)
    
    # Update 
    m = m - learning_rate * dm
    b = b - learning_rate * db
    
print("Slope(m) :", m)
print("Intercept(b) :", b)

# Predictions for Plotting 
y_pred = m*X + b

# Plot
plt.scatter(X, Y)
plt.plot(X, y_pred)
plt.xlabel("R&D Spend(Normalised)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()


```

## Output:
![linear regression using gradient descent](sam.png)

<img width="890" height="674" alt="image" src="https://github.com/user-attachments/assets/77666070-53e5-4efc-aa81-6803777f1de6" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
