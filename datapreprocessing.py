# MNIST data preprocessing to make it suitable for feeding into the model
# Written by Ashutosh Gupta


import pandas as pd
import numpy as np
from keras.utils import to_categorical

validation = 0.3

data = pd.read_csv('train.csv')
#print(data.describe())
data_X = np.array(data.drop(['label'], axis = 1))/255.0  #to scale
data_y = np.array(data['label']).reshape(-1, 1) #reshaped to one column per entry

length = int(len(data_X) * 0.2)
X_train = data_X[:-length]
y_train = to_categorical(data_y[:-length])
X_val = data_X[-length:]
y_val = to_categorical(data_y[-length:])

# print(X_val.shape)
# print(X_train.shape)
# print(y_train.head())

# X = np.reshape(X_train, (-1,1))
# print(X)

test = pd.read_csv('test.csv')
X_test = np.array(test)
# # print(X_test)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test', X_test)