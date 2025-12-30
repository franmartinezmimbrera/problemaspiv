#histogramagris.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  Calcular el histograma
hist = cv2.calcHist([gris], [0], None, [256], [0, 256])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(gris, cmap='gray')
ax1.set_title('Imagen en Grises')
ax1.axis('off')

ax2.plot(hist, color='black')
ax2.set_title('Histograma de Intensidad')
ax2.set_xlim([0, 256])
ax2.grid(True)

plt.tight_layout()
plt.show()
