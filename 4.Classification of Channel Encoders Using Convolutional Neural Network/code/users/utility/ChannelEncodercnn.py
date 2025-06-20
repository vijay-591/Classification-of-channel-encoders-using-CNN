import numpy as np
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from django.conf import settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def start_process():
    import pandas as pd
    path = os.path.join(settings.MEDIA_ROOT, "dataset_new.csv")
    data = pd.read_csv(path)
    # Load data from CSV file

    # Assume the last column is the target and the rest are features
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Generate dummy data for illustration purposes
    num_samples = 1000
    signal_length = 32  # Length of the encoded signals (adjust as necessary)
    num_classes = 4  # Number of different FEC schemes

    # Generate random data for the example
    # Reshape to (num_samples, signal_length, signal_length, 1) to simulate grayscale images
    X = np.random.rand(num_samples, signal_length, signal_length, 1)  # Example: 32x32 "images"
    y = np.random.randint(0, num_classes, num_samples)

    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes)

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
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    # y_pred = model.predict(X_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # cm = confusion_matrix(y_pred.round(), y_test)
    y_test_arg = np.argmax(y_test, axis=1)
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(y_test_arg, Y_pred)
    # rint(cm)
    # sns.heatmap(cm, annot=True)
    # plt.show()
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('accuracy_graph.png')
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('loss_graph.png')
    # plt.show()

    # Save the model (optional)
    model.save('cnn_fec_classifier.h5')
