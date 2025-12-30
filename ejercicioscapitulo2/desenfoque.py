#desenfoque.py
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('imagen.jpg')
kernel_size = (5, 5)
sigma = 1.4
img_desenfocada = cv2.GaussianBlur(img, kernel_size, sigma)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original (Nítida)')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(img_desenfocada, cv2.COLOR_BGR2RGB))
ax2.set_title(f'Desenfoque (Sigma={sigma})')
ax2.axis('off')
plt.tight_layout()
plt.show()
