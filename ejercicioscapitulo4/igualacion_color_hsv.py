#igualacion_color_hsv.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_equ = cv2.equalizeHist(v)

final_hsv = cv2.merge((h, s, v_equ))
final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original (Color)')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
ax2.set_title('Resultado (Ecualización en V)')
ax2.axis('off')

plt.tight_layout()
plt.show()
