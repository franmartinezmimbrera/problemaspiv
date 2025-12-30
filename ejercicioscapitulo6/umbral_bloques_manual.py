# umbral_bloques_manual.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
if img is None:
    print("Error: No se encuentra 'imagen.jpg'")
else:
    h, w = img.shape
    res = np.zeros_like(img)
    bs = 50 # Tamaño del bloque (block size)

    #  Procesamiento manual por bloques
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            # Extraer la región de interés (ROI)
            roi = img[i:i+bs, j:j+bs]
            
            # Aplicar Otsu localmente a este bloque
            # Otsu ignora el valor '0' y busca el umbral óptimo automáticamente
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Insertar el bloque procesado en la imagen de resultado
            res[i:i+bs, j:j+bs] = thresh

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Imagen Original (Gris)')
    ax1.axis('off')

    ax2.imshow(res, cmap='gray')
    ax2.set_title(f'Umbralizado por Bloques ({bs}x{bs})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
