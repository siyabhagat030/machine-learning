import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.linspace(0, 10, 100).reshape(-1, 1)
noise = np.random.normal(0, 1, x.shape)
y = 2 * x + 5 + noise
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, model.predict(x), color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
