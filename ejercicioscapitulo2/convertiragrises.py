#convertiragrises.py
import cv2
import matplotlib.pyplot as plt

# Leer la imagen original
img = cv2.imread('imagen.jpg')

#  Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Configurar el lienzo para 2 imágenes una al lado de la otra
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Convertir BGR de OpenCV a RGB de Matplotlib
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original (Color)')
ax1.axis('off')

ax2.imshow(gray_img, cmap='gray')
ax2.set_title('Escala de Grises')
ax2.axis('off')

plt.tight_layout() 
plt.show()
