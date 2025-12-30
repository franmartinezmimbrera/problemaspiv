#leermostrar.py
import cv2
import matplotlib.pyplot as plt
# Leer la imagen
img = cv2.imread('imagen.jpg')
# Mostrar la imagen
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Imprimir información sobre la imagen
print(f'Tamaño de la imagen: {img.shape}')
