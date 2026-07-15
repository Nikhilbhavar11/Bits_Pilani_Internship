import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset (handwritten numbers, downloads automatically)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Test on one image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Actual: {y_test[0]}, Predicted: {model.predict(x_test[:1]).argmax()}")
plt.show()