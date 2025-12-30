#igualacion_global.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: No se pudo cargar la imagen. Revisa el nombre del archivo.")
else:
    # Aplicar la Ecualización de Histograma Global
    equ = cv2.equalizeHist(img)
    # Usamos np.hstack para pegar las imágenes horizontalmente
    comparativa = np.hstack((img, equ))

    plt.figure(figsize=(12, 6))
    plt.imshow(comparativa, cmap='gray')   
    plt.title('Izquierda: Original | Derecha: Ecualización Global')
    plt.axis('off') 

    print(f"Tamaño de la imagen: {img.shape}")
    print("Mostrando comparativa... Cierra la ventana para terminar.")
    plt.show() 
