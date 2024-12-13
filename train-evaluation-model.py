import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

def train_evaluation_model(save_path='evaluation_model.h5'):
    # MNIST-Daten laden
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Daten vorverarbeiten
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]  # Hinzufügen der Kanaldimension
    x_test = x_test[..., tf.newaxis]
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Modell erstellen
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Modell trainieren
    model.fit(x_train, y_train, epochs=10, batch_size=128,
              validation_data=(x_test, y_test))
    
    # Modell speichern
    model.save(save_path)
    print(f"Evaluationsmodell gespeichert unter: {save_path}")

if __name__ == '__main__':
    train_evaluation_model()
