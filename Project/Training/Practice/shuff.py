import tensorflow as tf

data=tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
shuffled = data.shuffle(buffer_size=5,seed=42)

for item in shuffled:
    print(item.numpy())
    