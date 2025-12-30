#histograma_hsv_hue.py
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('imagen.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# El canal H (Hue) representa el color puro
hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])

plt.plot(hist_hue, color='orange')
plt.title('Histograma del Canal H (Matiz)')
plt.xlabel('Tono')
plt.ylabel('Frecuencia')
plt.show()
