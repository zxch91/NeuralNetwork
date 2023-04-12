import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_data = pd.read_excel('train_data.xlsx')
test_data = pd.read_excel('test_data.xlsx')

x_train = train_data[['T', 'W', 'SR', 'DSP', 'DRH']]
y_train = train_data['PanE']
x_test = test_data[['T', 'W', 'SR', 'DSP', 'DRH']]
y_test = test_data['PanE']

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Model')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)