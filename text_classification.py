import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load IMDB dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# GPU availability
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Fetch sample batch
train_example_batch, train_labels_batch = next(iter(train_data.batch(10).as_numpy_iterator()))
print(train_example_batch[:3])

# Load TensorFlow Hub embedding layer
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

# Build the model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Print model summary
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data.shuffle(10000).batch(100),
    epochs=25,
    validation_data=validation_data.batch(100).prefetch(tf.data.AUTOTUNE),
    verbose=1
)

# Evaluate the model
results = model.evaluate(test_data.batch(100), verbose=2)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.3f}")
