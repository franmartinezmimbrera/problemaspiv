#matriz_confusion.py
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Cargar datos (Dígitos 0-9)
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, shuffle=False
)

# Entrenar modelo (SVM)
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# Predecir
y_pred = clf.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 10))

# 'from_predictions' hace todo el trabajo duro:
# calcula la matriz y la dibuja con el mapa de color.
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    ax=ax, 
    cmap='Blues',      # Mapa de color azul profesional
    colorbar=True,     # Barra lateral de intensidad
    values_format='d'  # Mostrar números enteros
)

plt.title("Matriz de Confusión: Clasificación de Dígitos")
plt.ylabel('Etiqueta Real (Verdadera)')
plt.xlabel('Etiqueta Predicha (Modelo)')

plt.show()


cm = confusion_matrix(y_test, y_pred)

errores_3_como_8 = cm[3, 8] 

print("-" * 30)
print(f"ANÁLISIS DE ERROR ESPECÍFICO:")
print(f"Veces que el modelo vio un '3' pero predijo un '8': {errores_3_como_8}")
print("-" * 30)

