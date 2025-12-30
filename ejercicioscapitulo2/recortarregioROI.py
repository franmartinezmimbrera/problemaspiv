#recortarregioROI.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
#  Definir la región de interés (ROI)
x, y, w, h = 2000, 1400, 700, 700 
# 3. Recortar la ROI
roi = img[y:y+h, x:x+w]

img_con_cuadro = img.copy()
cv2.rectangle(img_con_cuadro, (x, y), (x+w, y+h), (0, 255, 0), 5) # Cuadro verde

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img_con_cuadro, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original (Zona marcada)')
ax1.axis('on') 

ax2.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
ax2.set_title(f'ROI: {w}x{h} píxeles')
ax2.axis('off')

plt.tight_layout()
plt.show()
