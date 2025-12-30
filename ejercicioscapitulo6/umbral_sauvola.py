# umbral_sauvola.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_sauvola # Importamos Sauvola

img_color = cv2.imread('imagentexto.png')
if img_color is None:
    print("Error: No se pudo cargar la imagen. Revisa el nombre y la extensión.")
else:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    #  Aplicar el umbral de Sauvola
    # window_size: tamaño del bloque (debe ser mayor que el grosor del texto)
    # k: sensibilidad (valor típico 0.2 para Sauvola)
    window_size = 25
    thresh = threshold_sauvola(img_gray, window_size=window_size, k=0.2)
    res = (img_gray > thresh).astype(np.uint8) * 255

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    ax1.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    ax1.set_title('Imagen Original (Color)')
    ax1.axis('off')

    ax2.imshow(res, cmap='gray')
    ax2.set_title(f'Método Sauvola (k=0.2, ws={window_size})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
