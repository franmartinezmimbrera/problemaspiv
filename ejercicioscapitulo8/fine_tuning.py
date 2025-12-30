import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import models, layers

# Supongamos que vamos a distinguir 3 cosas: "Piedra", "Papel", "Tijera"
NUM_CLASES = 3 
IMG_SHAPE = (224, 224, 3)

print(" Cargando el cerebro experto (MobileNetV2) ---")
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

# Congelar los pesos antiguos
# Esto evita que rompamos lo que la red ya ha aprendido durante años.
base_model.trainable = False

print(" Añadiendo capa personalizada ---")
model = models.Sequential([
    # El cerebro base
    base_model,    
    # Esta capa convierte el mapa de características 7x7x1280 en un solo vector de 1280
    layers.GlobalAveragePooling2D(), 
    # Capa intermedia para aprender mejor
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    # Capa de salida: Tantas neuronas como clases tengas
    layers.Dense(NUM_CLASES, activation='softmax') 
])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("¡Listo!")
