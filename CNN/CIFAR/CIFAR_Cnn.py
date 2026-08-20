import tensorflow as tf
from tensorflow.keras import datasets, layers,models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# print("Training Images Shape:", X_train.shape)          ## (50000,32,32,3)
# print("Training Labels Shape:", y_train.shape)          ## (50000,1)
# print("Testing Images Shape:", X_test.shape)            ## (10000,32,32,3)
# print("Testing Labels Shape:", y_test.shape)            ## (10000,1)

# plt.imshow(X_train[1000])
# plt.show()

# plt.figure(figsize=(15,2))
# plt.imshow(X_train[1000])
# plt.show()

# print(y_train[:5])

y_train = y_train.reshape(-1,)

# print(y_train.shape)
# print(y_train[:5])

images_classes = ["Plane", "Automobile", "Bird", "Cat", "Deer","Dog", "Frog", "Horse", "Ship", "Truck"]

# # print(images_classes[5])

# def plot_classes(X,y,st,en):
#     for i in range(st,en):
#         plt.figure(figsize=(15,2))
#         plt.imshow(X[i])
#         plt.xlabel(images_classes[y[i]])
#     plt.show()
# plot_classes(X_train,y_train,1000,1005)


X_train = X_train/255
X_test = X_test/255

# ann = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(32, 32, 3)), 
#     tf.keras.layers.Dense(3000, activation='relu'),     
#     tf.keras.layers.Dense(1000, activation='relu'), 
#     tf.keras.layers.Dense(10, activation='softmax')    
# ])

# ann.compile(
#     optimizer='SGD',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# ann.fit(X_train,y_train,epochs=5)

# ann.evaluate(X_test,y_test)
# y_pred_ann=ann.predict(X_test)
# y_pred_classes_ann=[np.argmax(res) for res in y_pred_ann]
# print(classification_report(y_test,y_pred_classes_ann))

cnn = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_pred_cnn = cnn.predict(X_test)
y_pred_classes_cnn = [np.argmax(i) for i in y_pred_cnn]

print(classification_report(y_test, y_pred_classes_cnn))