#reflejarimagen.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')

img_reflejada_h = cv2.flip(img, 1)
img_reflejada_v = cv2.flip(img, 0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(img_reflejada_h, cv2.COLOR_BGR2RGB))
ax2.set_title('Horizontal (flipCode=1)')
ax2.axis('off')

ax3.imshow(cv2.cvtColor(img_reflejada_v, cv2.COLOR_BGR2RGB))
ax3.set_title('Vertical (flipCode=0)')
ax3.axis('off')

plt.tight_layout()
plt.show()
