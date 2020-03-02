import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#get data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data
(x_train, y_train), (x_test, y_test) = (tf.keras.utils.normalize(x_train, axis=1), y_train), (tf.keras.utils.normalize(x_test, axis=1), y_test)

new_model = tf.keras.models.load_model("num_reader.model")
predictions = new_model.predict(x_test)

for i in range(0,2,1):
  print("The prediction is: ")
  print(np.argmax(predictions[i]))
  print("The image is: ")
  plt.imshow(x_test[i])
  plt.show()
