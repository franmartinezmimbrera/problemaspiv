#deteccion_exposicion.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

total_píxeles = img.shape[0] * img.shape[1]
subexpuestos = (hist[0] / total_píxeles) * 100
sobreexpuestos = (hist[255] / total_píxeles) * 100

print(f"Porcentaje píxeles negros: {subexpuestos[0]:.2f}%")
print(f"Porcentaje píxeles blancos: {sobreexpuestos[0]:.2f}%")

if subexpuestos > 5: print("Alerta: Posible imagen subexpuesta")
if sobreexpuestos > 5: print("Alerta: Posible imagen sobreexpuesta (quemada)")
