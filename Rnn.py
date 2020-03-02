import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train.shape is 60000, 28, 28 images

# normalize data
(x_train, y_train), (x_test, y_test) = (tf.keras.utils.normalize(x_train, axis=1), y_train), (tf.keras.utils.normalize(x_test, axis=1), y_test)

neural = tf.keras.models.Sequential()  # sequential type of model

neural.add(LSTM(128,input_shape=(28, 28), activation=tf.nn.relu, return_sequences = True))
neural.add(Dropout(0.2))

neural.add(LSTM(128, activation=tf.nn.relu))
neural.add(Dropout(0.1))

neural.add(Dense(32, activation=tf.nn.relu))
neural.add(Dropout(0.1))

neural.add(Dense(10, activation=tf.nn.softmax))

optm = Adam(lr=1e-3, decay=1e-3)
neural.compile(optimizer=optm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

neural.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

model.save("num_reader.model")
