# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#  Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

# Predicting a new result
print(regressor.predict([[6.5]]))

# Visualising the Decision Tree Regression results (higher resolution)
# only for this case, not for multi dimensional features
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, edgecolors='red')
plt.plot(x_grid, regressor.predict(x_grid ), color='blue')
plt.title('Truth or bluff (Decision Tree Regression)')
plt.ylabel('Salary')
plt.show()