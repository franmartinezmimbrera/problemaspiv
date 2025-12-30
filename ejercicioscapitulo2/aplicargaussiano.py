#aplicargaussiano.py
import cv2
import matplotlib.pyplot as plt

#  Leer la imagen
img = cv2.imread('imagen.jpg')

# 2. Aplicar filtro gaussiano. (35, 35) es el tamaño del kernel (debe ser impar). A mayor número, más desenfoque.
gaussian_blur = cv2.GaussianBlur(img, (35, 35), 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original (Nítida)')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
ax2.set_title('Desenfoque Gaussiano (Kernel 35x35)')
ax2.axis('off')

plt.tight_layout()
plt.show()
