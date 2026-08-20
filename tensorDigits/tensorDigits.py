import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


# print(tf.__version__)
# print(keras.__version__)

## DATA PREPROCESSING - STARTS
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
# print(X_train.shape)           ## (60000, 28, 28)
# print(y_train.shape)           ## (60000,)
# print(X_test.shape)            ## (10000, 28, 28)
# print(y_test.shape)            ## (10000,)

# plt.matshow(X_train[1])
# plt.show()

# print(y_train[1])
# print(y_train[0])

X_train= X_train/255
# print(X_train[0])

X_test = X_test/255
# print(X_test[0])

X_train_nn = X_train.reshape(len(X_train),28*28)
print(X_train_nn.shape)          ## (60000, 784)

X_test_nn = X_test.reshape(len(X_test),28*28)
print(X_test_nn.shape)           ## (10000, 784)

## DATA PREPROCESSING - ENDS

## MODEL TRAINING - STARTS

model= tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=(784,),activation="sigmoid")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_nn, y_train, epochs=5)

model.evaluate(X_test_nn, y_test)

y_train_pred = model.predict(X_train_nn)
# print(y_train_pred[2])
# plt.matshow(X_train[2])
# plt.show()

# print(y_train_pred[2].argmax())        ## argmax give largest value in an array. 
# print(y_train_pred.max())

y_test_pred = model.predict(X_test_nn)
# print(y_test_pred[0])
# plt.matshow(X_test[0])
# plt.show()
# print(y_test_pred[0].argmax())


y_test_predicted = [np.argmax(X) for X in y_test_pred]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_test_predicted)
print(cm)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True,fmt='d')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show()



