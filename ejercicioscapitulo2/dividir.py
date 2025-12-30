#dividir.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')

b, g, r = cv2.split(img)

plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(b, cmap='Blues'); plt.axis('off')
plt.subplot(132); plt.imshow(g, cmap='Greens'); plt.axis('off')
plt.subplot(133); plt.imshow(r, cmap='Reds'); plt.axis('off')
plt.show()
