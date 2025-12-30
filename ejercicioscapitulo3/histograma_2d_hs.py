#histograma_2d_hs.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Calculamos histograma para H (0) y S (1) simultáneamente
hist2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

plt.imshow(hist2d, interpolation='nearest', origin='lower', extent=[0, 256, 0, 180])
plt.colorbar(label='Frecuencia')
plt.xlabel('Saturación')
plt.ylabel('Matiz (Hue)')
plt.title('Histograma 2D H-S')
plt.show()
