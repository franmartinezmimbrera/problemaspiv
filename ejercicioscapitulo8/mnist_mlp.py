#mnist_mlp.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Limpiar logs rojos
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
# Carga de datos
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  Normalización (0-255 -> 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0
#  Modelo 
model = Sequential([
    Input(shape=(28, 28)),          # Capa de entrada explícita
    Flatten(),                  # Aplana la imagen 28x28 a vector de 784
    Dense(128, activation='relu'),  # Capa oculta
    Dense(10, activation='softmax')]) # Capa de salida (10 números)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Iniciando entrenamiento...")
model.fit(x_train, y_train, epochs=5)
print("\n Evaluando en datos de test ....")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrecisión final en Test: {test_acc*100:.2f}%")
