# Best: convert ot jupyter notebook and run each cell. 
# cells are delimited ===

# Cell 0 =====
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cell 1 =====
DATA_SOURCE = '../data/FuelConsumptionCo2.csv'

df = pd.read_csv(DATA_SOURCE)

# Cell 2 =====
df.head()

# Cell 3 =====
df.describe()

# Cell 4 =====
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# Cell 5 =====
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
 
# Cell 6 =====
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# Cell 7 =====
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Cell 8 =====
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# Cell 9 =====
# Train and test
# split dataset to train and test,  both mutually exclusive.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data
regr = linear_model.LinearRegression()

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Model data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual Sum of Squares (MSE): %.2f" %
      np.mean(np.absolute(test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# ===== FUELCONSUMPTION_COMB
train_x = train[['FUELCONSUMPTION_COMB']]
test_x = test[['FUELCONSUMPTION_COMB']]
regr.fit(train_x, train_y)
predictions = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))
