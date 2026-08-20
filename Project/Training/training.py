# import os
# print("Current Working Directory",os.getcwd())
# print("Dataset Exits : ",os.path.exists("./Dataset"))
# print("Folders :",os.listdir("./Dataset"))

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers

# print(tf.keras.preprocessing.image_dataset_from_directory)

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory("./Dataset",shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE),batch_size=BATCH_SIZE)
# print(dataset)
# print(len(dataset))

class_names = dataset.class_names
# print("\n\n\nOutput : ",class_names)

# i=0
# num_of_iamge_batches=0
# total_number_of_images =0
# for image_batch, label_batch in dataset:
#     print("\nOutput")
#     print("Batch Number : ",i)
#     i=i+1
#     print(image_batch.shape)
#     print(label_batch.numpy())
#     print(len(label_batch.numpy()))
#     num_of_iamge_batches += 1
#     total_number_of_images += image_batch.shape[0]
    
# print("Total Number of Image Batches :",num_of_iamge_batches)
# print("Total Number of  Images :" ,total_number_of_images)

# i = 0
# for image_batch, label_batch in dataset.take(1):
#     # print(image_batch[0].shape)           ## (256, 256, 3)
#     # print(image_batch[0])
#     # print(image_batch[0].numpy())
#     i=i+1
#     print("I = ",i)
#     plt.figure(figsize=(15,15))
#     for i in range(0,15):
#         plt.subplot(3, 5, i + 1)
#         plt.imshow(image_batch[i].numpy().astype('uint8'))
#         # plt.title(i)
#         plt.title(class_names[label_batch[i]])
#         plt.axis("off")
#     plt.show()

train_size = 0.8
total_training_images = np.round(len(dataset)*train_size)
# print(total_training_images)               ## 113.0

train_ds = dataset.take(total_training_images)
# print(len(train_ds))                       ## 113

total_val_test_images = dataset.skip(total_training_images)
# print(len(total_val_test_images))          ## 28

val_size = 0.1
total_vali_images = np.round(len(dataset)*val_size)
# print(total_vali_images)                   ## 14.0

vail_ds = total_val_test_images.take(total_vali_images)
# print(len(vail_ds))                        ## 14

test_size = 0.1
total_testing_images = np.round(len(dataset)*test_size)
# print(total_testing_images)                ## 14.0

test_ds = total_val_test_images.take(total_vali_images)
# print(len(test_ds))                        ## 14

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=12)
        
    train_size = np.round(train_split*ds_size)
    train_ds = ds.take(train_size)
    val_size=np.round(val_split*ds_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).take(val_size)
    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds = get_dataset_partitions_tf(dataset)

# print("Training Batches   :", len(train_ds))
# print("Validation Batches :", len(val_ds))
# print("Testing Batches    :", len(test_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 3

model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(n_classes,activation='softmax')   
])

model.build(input_shape=input_shape)
model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

history = model.fit(
    train_ds,
    batch_size = BATCH_SIZE,
    validation_data = val_ds,
    verbose = 1,
    epochs = 5
)

scores = model.evaluate(test_ds)
print(scores)

print(history.params)
print(history.history.keys())
print(type(history.history['loss']))
print(len(history.history['loss']))
print(history.history['loss'][:5])
acc = history.history['accuracy']
print(acc)
val_acc = history.history['val_accuracy']
print(val_acc)
loss = history.history['loss']
print(loss)
val_loss = history.history['val_loss']
print(val_loss)

for images_batch,labels_batch in test_ds.take(1):
    first_image=images_batch[0].numpy().astype('uint8')
    first_label=labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    plt.show()
    print("actual label : ",class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("Predicted label : ",class_names[np.argmax(batch_prediction[0])])
      
def predict(model,image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array,0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.show()
        predicted_class, confidence = predict(model,images[i].numpy())
        actual_class = class_names[labels[i]]
        
        plt.title(f"Actual : {actual_class}, \n Predicted: {predicted_class}, \n Confidence: {confidence}")
        plt.axis('off')