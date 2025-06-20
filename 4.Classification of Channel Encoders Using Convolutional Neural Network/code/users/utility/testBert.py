import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Generate dummy data for illustration purposes
num_samples = 1000
signal_length = 128  # Length of the encoded signals
num_classes = 4  # Number of different encoding schemes

# Generate random data for the example
X = np.random.rand(num_samples, signal_length, signal_length, 1)  # Example: grayscale images
y = np.random.randint(0, num_classes, num_samples)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes)
print(X)
print("======================================")
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(signal_length, signal_length, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Save the model (optional)
model.save('cnn_blind_encoder_classifier.h5')
