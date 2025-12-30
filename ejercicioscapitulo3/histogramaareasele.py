#histogramaareasele.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
# Definir una región de interés (ROI)
roi = img[100:300, 150:400]
# Calcular el histograma del ROI
hist_roi = cv2.calcHist([roi], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histograma de la ROI")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist_roi)
plt.xlim([0, 256])
plt.show()
