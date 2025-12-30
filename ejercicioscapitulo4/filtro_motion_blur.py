#filtro_motion_blur.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagenbordes.jpg', 0)

size = 15
kernel_motion = np.zeros((size, size))
kernel_motion[int((size-1)/2), :] = np.ones(size)
kernel_motion = kernel_motion / size

res = cv2.filter2D(img, -1, kernel_motion)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')
ax2.imshow(res, cmap='gray')
ax2.set_title(f'Motion Blur Horizontal ({size}px)')
ax2.axis('off')

plt.tight_layout()
plt.show()
