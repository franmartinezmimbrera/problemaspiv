from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

# BINARIO (Perro vs Gato) 
# Generamos datos dummy: 2 clases (0 y 1)
X_bin, y_bin = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=1)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bin, y_bin, stratify=y_bin)

# Definición del modelo Binario
modelo_binario = MLPClassifier(
    hidden_layer_sizes=(10,), 
    activation='relu',       # Activación oculta
    solver='adam',
    max_iter=500,
    random_state=1
)

modelo_binario.fit(X_train_b, y_train_b)

print(" MODELO BINARIO")
print(f"Función de Activación de Salida (Implícita): Logistic (Sigmoid)")
print(f"Función de Pérdida usada: Log-Loss (Binary Crossentropy)")
print(f"Predicción (probabilidad clase 1): {modelo_binario.predict_proba(X_test_b[:1])[0][1]:.4f}")

#  MULTI-CLASE (Dígitos 0-9) 
# Generamos datos dummy: 3 clases (0, 1, 2)
X_multi, y_multi = make_blobs(n_samples=100, centers=3, random_state=1)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi)

# Definición del modelo Multi-clase
modelo_multiclase = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=1
)

modelo_multiclase.fit(X_train_m, y_train_m)

print("\nMODELO MULTI-CLASE")
print(f"Función de Activación de Salida (Implícita): Softmax")
print(f"Función de Pérdida usada: Cross-Entropy Loss")
probs = modelo_multiclase.predict_proba(X_test_m[:1])[0]
print(f"Predicción (vector de probs): {probs}")
print(f"Suma de probabilidades: {np.sum(probs):.2f}") 
