import numpy as np
import pickle as pkl
import cv2
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import Sequential
from digit_ocr_model.create_dataset import NumberDataSet


data_size = 70000
batch_size = 200
image_dim = 64

num_data_Set = NumberDataSet((data_size // batch_size), batch_size, image_dim)
# Create model
# Building CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_dim, image_dim, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

save_call = ModelCheckpoint("digit_model.h5", monitor='val_loss', verbose=0, save_best_only=False,
                            save_weights_only=False, mode='auto', period=1)

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Started training')
model.fit(num_data_Set, callbacks=[save_call], epochs=500)

model.save("digit_model.h5")
