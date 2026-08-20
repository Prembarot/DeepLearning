import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets , layers, models

##  READING CSV
input_train_path = "Dataset/input.csv"
X_train = np.loadtxt(input_train_path, delimiter=',')
label_train_path = "Dataset/labels.csv"
y_train = np.loadtxt(label_train_path, delimiter=',')
input_test_path = "Dataset/input_test.csv"
X_test = np.loadtxt(input_test_path, delimiter=',')
label_test_path = "Dataset/labels_test.csv"
y_test = np.loadtxt(label_test_path, delimiter=',')

## RESHAPE
X_train = X_train.reshape(len(X_train),100,100,3)
y_train = y_train.reshape(len(y_train),1)
X_test = X_test.reshape(len(X_test),100,100,3)
y_test = y_test.reshape(len(y_test),1)

## SCALLING
X_train = X_train/255
X_test = X_test/255

## VISUALIZATION
# for i in range(1500,1510):
#     plt.imshow(X_train[i,:])
#     plt.show()
    
## CREATING CON2D & neural network 
cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

## compile model
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

## TRAINING MODEL
cnn.fit(X_train,y_train,epochs=10)

## Evaluation 
cnn.evaluate(X_test,y_test)

plt.imshow(X_test[169,:])
plt.show()

y_pred = cnn.predict(X_test[169,:].reshape(1,100,100,3))
print(y_pred)

if (y_pred >= 0.5):
    print("Cat")
else:
    print("Dog")