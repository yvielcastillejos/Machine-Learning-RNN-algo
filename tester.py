import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model("num_reader.model")
predictions = new_model.predict(x_test)

#prints 2 predictions of the images
for i in range(0,2,1):
  print("The prediction is: " + string(np.argmax.predictions[i]))
  print("The image is: ")
  plt.imshow(x_test[i])
