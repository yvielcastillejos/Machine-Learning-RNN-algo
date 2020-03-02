import matplotlib.pyplot as plt
import numpy as np

#save model

#print the accuracy and loss
value_loss, value_acc = model.evaluate(x_test,y_test)
print(value_loss,value_acc) #too much means underfit, too little overfit

new_model = tf.keras.models.load_model("num_reader.model)
predictions = new_model.predict(x_test)

#prints 2 predictions of the images
for i in range(0,2,1):
  print("The prediction is: " + string(np.argmax.predictions[i]))
  print("The image is: ")
  plt.imshow(x_test[i])
