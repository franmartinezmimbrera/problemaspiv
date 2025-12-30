#rotar.py
import cv2
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread('imagen.jpg')
# Obtener las dimensiones de la imagen
(h, w) = img.shape[:2]

# Calcular el centro de la imagen
center = (w // 2, h // 2)

# Generar la matriz de rotación
M = cv2.getRotationMatrix2D(center, 90, 1.0)

# Realizar la rotación
rotated_img = cv2.warpAffine(img, M, (w, h))

# Mostrar la imagen rotada
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

