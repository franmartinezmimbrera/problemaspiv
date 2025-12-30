import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Limpieza de logs

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import matplotlib.pyplot as plt

# Nombres de las clases (Para que sea legible)
class_names = ['Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
# Carga y Preprocesamiento
print("Cargando Fashion MNIST...-")
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
# Normalizar (0-1) y Reshape (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# Definición del Modelo (CNN de dos niveles)
model = models.Sequential([
    Input(shape=(28, 28, 1)),
    
    # Bloque 1: Detecta bordes y formas básicas
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Bloque 2: Detecta patrones más complejos (texturas, formas de ropa)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Clasificación
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(" Entrenando...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
# Elegimos una imagen aleatoria del test
idx = np.random.randint(0, len(x_test))
imagen = x_test[idx]
etiqueta_real = y_test[idx]
#Predicción
prediccion_raw = model.predict(np.expand_dims(imagen, axis=0), verbose=0)
clase_predicha = np.argmax(prediccion_raw)
print(f"\n Resultado de la prueba...")
print(f"Realidad:   {class_names[etiqueta_real]}")
print(f"Predicción: {class_names[clase_predicha]}")

plt.figure(figsize=(3,3))
plt.imshow(imagen.reshape(28, 28), cmap='gray')
plt.title(f"IA dice: {class_names[clase_predicha]}")
plt.axis('off')
plt.show()
