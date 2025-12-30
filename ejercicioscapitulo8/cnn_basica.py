#cnn_basica.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Silenciar logs de TensorFlow

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

# Preparar datos (MNIST)
# Las CNN necesitan 3 dimensiones: (Alto, Ancho, Canales). 
# MNIST viene como (28, 28), así que añadimos el "1" del canal (escala de grises).
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Definición del Modelo 
model = models.Sequential([
    # Capa de Entrada Explícita: (Alto, Ancho, Canales de color)
    Input(shape=(28, 28, 1)),  
    # Bloque de Convolución
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Aplanado y Clasificación
    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dense(10, activation='softmax')
])

#Resumen y Compilación
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
print("\n-Entrenando CNN....")
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)
