#histograma_lab.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(lab)

plt.figure(figsize=(10,4))
plt.subplot(131), plt.plot(cv2.calcHist([l],[0],None,[256],[0,256])), plt.title('L (Luminosidad)')
plt.subplot(132), plt.plot(cv2.calcHist([a],[0],None,[256],[0,256])), plt.title('a (Verde-Rojo)')
plt.subplot(133), plt.plot(cv2.calcHist([b],[0],None,[256],[0,256])), plt.title('b (Azul-Amarillo)')
plt.tight_layout()
plt.show()
