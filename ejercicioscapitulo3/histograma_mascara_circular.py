#histograma_mascara_circular.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]

# Crear máscara circular
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, (w//2, h//2), min(w, h)//4, 255, -1)

hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(121), plt.imshow(cv2.bitwise_and(img, img, mask=mask), cmap='gray')
plt.subplot(122), plt.plot(hist_mask)
plt.show()
