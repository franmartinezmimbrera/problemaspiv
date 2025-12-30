#data_augmentation.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
# Cargamos una imagen de ejemplo (Fashion MNIST)
(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
# Cogemos una imagen (por ejemplo, la número 9, que suele ser un zapato o bolso)
imagen_original = x_train[9]
# Le damos forma: (1 imagen, 28 alto, 28 ancho, 1 canal)
imagen_input = imagen_original.reshape((1, 28, 28, 1)).astype('float32') / 255.0

datagen = ImageDataGenerator(
    rotation_range=20,      # Rotar hasta 20 grados
    zoom_range=0.15,        # Zoom de hasta 15%
    horizontal_flip=True,   # Voltear horizontalmente (espejo)
    fill_mode='nearest'     # Cómo rellenar los huecos al rotar
)
# Vamos a generar 5 variantes de esa misma imagen
print("Generando imágenes aumentadas...")
iterador = datagen.flow(imagen_input, batch_size=1)

plt.figure(figsize=(10, 4))
# Mostramos la original
plt.subplot(1, 6, 1)
plt.title("Original")
plt.imshow(imagen_original, cmap='gray')
plt.axis('off')
# Mostramos 5 transformaciones generadas por datagen
for i in range(5):
    plt.subplot(1, 6, i + 2)
    batch = next(iterador) 
    imagen_generada = batch[0].reshape(28, 28)
    plt.title(f"Gen {i+1}")
    plt.imshow(imagen_generada, cmap='gray')
    plt.axis('off')
plt.show()
