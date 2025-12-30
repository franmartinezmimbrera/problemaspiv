#prevencion_overfitting.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Limpiar logs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten

# La entrada son imágenes aplanadas de 28x28 = 784 píxeles.
# Si fueran imágenes sin aplanar, usaríamos Input(shape=(28, 28)) y luego Flatten().

# Modelo
model = Sequential([
    #  Capa de entrada explícita 
    Input(shape=(784,)),

    #  Primera capa densa 
    Dense(512, activation='relu'),

    # 3. Capa de Dropout agen)
    # Durante el entrenamiento, "apaga" aleatoriamente el 50% de las neuronas
    # de la capa anterior en cada paso. Esto fuerza a la red a no depender
    # de ninguna neurona específica, reduciendo el sobreajuste.
    Dropout(0.5),

    # Capa de salida (como en la imagen)
    Dense(10, activation='softmax')
])

print("Resumen del Modelo con Dropout")
model.summary()
