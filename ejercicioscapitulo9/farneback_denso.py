#farneback_denso.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

if img1 is None or img2 is None:
    print("Error: No se encontraron las imágenes.")
    sys.exit()

# Redimensionar si es necesario (Farneback es pesado, mejor imágenes no gigantes)
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convertir a escala de grises
prvs = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
next_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ALGORITMO FARNEBACK 
# Calcula el flujo para TODOS los píxeles
# flow tendrá dimensiones (alto, ancho, 2). El '2' son los vectores (dx, dy)
flow = cv2.calcOpticalFlowFarneback(prvs, next_img, None, 
                                    pyr_scale=0.5, levels=3, winsize=15, 
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

#  PREPARAR VISUALIZACIÓN HSV 
# Creamos una imagen vacía con 3 canales (Hue, Saturation, Value)
hsv = np.zeros_like(img1)

# Ponemos la Saturación al máximo (255) para colores vivos
hsv[..., 1] = 255

# CONVERSIÓN CARTESIANA A POLAR 
# Convertimos los vectores (x, y) a (Magnitud, Ángulo)
# mag: Qué tan rápido se mueve
# ang: Hacia dónde se mueve
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#  El ÁNGULO determina el COLOR (Hue)
# Convertimos radianes a grados (OpenCV usa 0-180 para Hue)
hsv[..., 0] = ang * 180 / np.pi / 2

#  La MAGNITUD determina el BRILLO (Value)
# Normalizamos para que se vea bien (0 a 255)
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

# Convertir de HSV a BGR para mostrarlo
bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Frame 1 (Original)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(bgr_flow, cv2.COLOR_BGR2RGB))
plt.title("Flujo Denso (Farneback)\nColor=Dirección, Brillo=Velocidad")
plt.axis('off')

plt.tight_layout()
plt.show()
