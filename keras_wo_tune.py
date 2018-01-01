import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import seaborn as sns

# data processing
df = pd.read_csv('/Users/Xueyao/Documents/GitHub/fer2013.csv')

# split training data
train = df[['emotion','pixels']][df['Usage'] == 'Training']
train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train['emotion'])
print(x_train.shape, y_train.shape)

# split test data
public_test_df = df[['emotion', 'pixels']][df['Usage'] == 'PublicTest']
public_test_df['pixels'] = public_test_df['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df['pixels'].values)
y_test = np.array(public_test_df['emotion'])
print(x_test.shape, y_test.shape)

# reshape training data to N * 48 * 48 * 1
x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)
print(x_train.shape, x_test.shape)

# one hot encoding y_train, y_test to indicators
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape, y_test.shape)

# show some facial expressions
plt.figure(0, figsize=(12, 6))
for i in range(1, 13):
    plt.subplot(3, 4, i)
    plt.imshow(x_train[i, :, :, 0], cmap='gray')
plt.tight_layout()
plt.show()

# build keras model
model = Sequential()

# 1st conv layer
model.add(Conv2D(64, 3, data_format='channels_last', kernel_initializer='he_normal', input_shape=(48, 48, 1)))
model.add(BatchNormalization()) # try put BatchNormalization after Activation
model.add(Activation('relu'))

# 2nd convpool layer
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6)) # tune dropout rate

# 3rd conv layer
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 4th conv layer
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5th convpool layer
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6)) # tune dropout rate

# 6th feed forward layer
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6)) #tune

# output classification layer
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# save best weights
checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)

# number of epochs
epochs = 10

# train model
hist = model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                batch_size=100, validation_data=(x_test, y_test),
                callbacks=[checkpointer], verbose=2)

# save model to json
model_json = model.to_json()
with open('face_model.json', 'w') as json_file:
    json_file.write(model_json)


# plot the training loss, validation loss, training accuracy, validation accuracy
plt.figure(figsize=(14, 3))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer: Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], color='b', label='Training Loss')
plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# process provate test data
test = df[['emotion', 'pixels']][df['Usage'] == 'PrivateTest']
test['pixels'] = test['pixels'].apply(lambda im: np.fromstring(im, sep=' '))

x_test_private = np.vstack(test['pixels'].values)
y_test_private = np.array(test['emotion'])
x_test_private = x_test_private.reshape(-1, 48, 48, 1)
y_test_private = np.utils.to_categorical(y_test_private)

# evaluate model on private test data
score = model.evaluate(x_test_private, y_test_private, verbose=0)
score





