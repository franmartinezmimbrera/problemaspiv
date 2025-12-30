import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import models, layers, Input

print("Descargando datos MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("--- 2. Creando la red neuronal... ---")
model = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando...")

model.fit(x_train, y_train, epochs=5, batch_size=64) 

print(" Guardando archivo...")
nombre_archivo = 'mi_red_mnist.keras'
model.save(nombre_archivo)

# Verificación final
if os.path.exists(nombre_archivo):
    print(f"\n El archivo '{nombre_archivo}' ha sido creado.")
else:
    print("\n Error al guardar el archivo.")
