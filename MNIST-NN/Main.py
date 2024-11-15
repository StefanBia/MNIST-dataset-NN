import tensorflow as tf
import numpy as np

# Updated Model Using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28*28,)),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using a more advanced optimizer and loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Loading and preparing the MNIST data using tf.data for efficiency
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255

# Use tf.data for efficient batch processing
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

# Training the model
model.fit(train_dataset, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.2f}")

# Make predictions and calculate accuracy
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"Accuracy: {matches.mean():.2f}")
