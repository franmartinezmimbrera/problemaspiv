#histograma_rgb_superpuesto.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
colores = ('b', 'g', 'r')

for i, col in enumerate(colores):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col, alpha=0.7, label=f'Canal {col.upper()}')
    plt.fill_between(range(256), hist.flatten(), color=col, alpha=0.1)

plt.title('Histogramas RGB Superpuestos')
plt.legend()
plt.show()
