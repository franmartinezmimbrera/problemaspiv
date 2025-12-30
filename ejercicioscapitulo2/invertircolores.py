#invertircolores.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
# La función bitwise_not invierte cada bit de cada píxel.
img_invertida = cv2.bitwise_not(img)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(img_invertida, cv2.COLOR_BGR2RGB))
ax2.set_title('Invertida (Negativo)')
ax2.axis('off')

plt.tight_layout()
plt.show()
