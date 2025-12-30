#estelas.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# CARGA DE IMÁGENES 
nombre_img1 = 'img1.jpg'
nombre_img2 = 'img2.jpg'

img1 = cv2.imread(nombre_img1)
img2 = cv2.imread(nombre_img2)

if img1 is None or img2 is None:
    print(f"Error: No se pudieron cargar {nombre_img1} o {nombre_img2}.")
    print("Verifica que los archivos estén en la carpeta correcta.")
    sys.exit()

if img1.shape != img2.shape:
     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#  Detectar características en la primera imagen (Shi-Tomasi)
p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)

# Calcular Flujo Óptico (Lucas-Kanade)
# Calcula la nueva posición (p1) en la segunda imagen
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

vis = img2.copy() 

if p1 is not None:
    # Seleccionar solo los puntos encontrados con éxito (status = 1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # Posición actual (img2)
        c, d = old.ravel() # Posición anterior (img1)
        
        # Línea ROJA: Vector de desplazamiento (desde viejo a nuevo)
        cv2.line(vis, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
        # Punto VERDE: Ubicación actual en img2
        cv2.circle(vis, (int(a), int(b)), 5, (0, 255, 0), -1)
else:
    print("No se pudo calcular el flujo óptico para ningún punto.")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1) # (Filas, Columnas, Índice)
# Convertir BGR a RGB para matplotlib
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Frame 1 (Origen: t)')
plt.axis('off') 

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Frame 2 (Destino: t+1)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title('Resultado Lucas-Kanade')
plt.axis('off')

plt.tight_layout()
plt.show()
