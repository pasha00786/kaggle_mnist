# MNIST data preprocessing to make it suitable for feeding into the model
# Written by Ashutosh Gupta


import pandas as pd
import numpy as np


data = pd.read_csv('train.csv')
#print(data.describe())
X_train = np.array(data.drop(['label'], axis = 1))
y_train = np.array(data['label'])
# print(y_train.head())

# X = np.reshape(X_train, (-1,1))
# print(X)

test = pd.read_csv('test.csv')
X_test = np.array(test)
# print(X_test)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test', X_test)