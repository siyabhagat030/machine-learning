import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.show()

