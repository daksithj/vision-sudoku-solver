import numpy as np
import pickle as pkl
import cv2
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.python.keras import Sequential

X_train = []
y_train = []

train_data_size = 60000
test_data_size = 10000

with open('number_list.pkl', 'rb') as f:
    num_list = pkl.load(f)

for a in range(train_data_size):
    image = cv2.imread(f'ocr_dataset/{a}.png', cv2.IMREAD_GRAYSCALE)
    X_train.append(image)
    y_train.append(num_list[a][1])


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = []
y_test = []

for a in range(train_data_size, train_data_size + test_data_size):
    image = cv2.imread(f'output/{a}.png', cv2.IMREAD_GRAYSCALE)
    X_test.append(image)
    y_test.append(num_list[a][1])


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# convert from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0


# Create model
# Building CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=200)

model.save("digit_model.h5")