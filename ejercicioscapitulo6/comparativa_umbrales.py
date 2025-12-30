# comparativa_umbrales.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)

if img is None:
    print("Error: No se pudo cargar 'imagen.jpg'.")
else:
    # Umbral Global Simple (valor fijo de 127)
    _, t1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Umbral Adaptativo Gaussiano
    # Calcula el umbral para áreas pequeñas de 11x11 píxeles.
    t2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original (Gris)')
    ax1.axis('off')


    ax2.imshow(t1, cmap='gray')
    ax2.set_title('Global (127)')
    ax2.axis('off')


    ax3.imshow(t2, cmap='gray')
    ax3.set_title('Adaptativo Gaussiano')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
