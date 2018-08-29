# Keras model for digit recogniser
# Written by Ashutosh Gupta

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import pandas as pd

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')

# # print(X_train)
# # print(y_train)

# # y_binary = to_categorical(y_train)
# # print(y_binary)

model = Sequential()

model.add(Dense(64, activation = 'relu', input_dim = 784))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile( optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 32)
scores = model.evaluate(X_val, y_val)
print(scores[1])


# Once the prediction is done, test prediction  written to a CSV file.
results= model.predict(X_test)
np.save('results.npy', results)

results = np.load('results.npy')
Label = []
for result in results:
	Label.append(np.argmax(result))

ImageId = np.arange(1,len(Label)+1)

df = pd.DataFrame({'ImageId':ImageId, 'Label':Label})
df.to_csv('submission_new.csv', index = False)