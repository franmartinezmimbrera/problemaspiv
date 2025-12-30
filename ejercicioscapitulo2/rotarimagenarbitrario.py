#rotarimagenarbitrario.py
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('imagen.jpg')
# Configurar la rotación
angulo = 45  # grados (sentido antihorario)
(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2) # Centro de la imagen
# Calcular la matriz de rotación 2D
M = cv2.getRotationMatrix2D((cX, cY), angulo, 1.0)
# Aplicar la transformación (Warp Affine)
img_rotada = cv2.warpAffine(img, M, (w, h))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(img_rotada, cv2.COLOR_BGR2RGB))
ax2.set_title(f'Rotada {angulo} grados')
ax2.axis('off')
plt.tight_layout()
plt.show()
