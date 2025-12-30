#recortar.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
alto, ancho, _ = img.shape

# Calcular el punto central de la imagen
centro_y = alto // 2
centro_x = ancho // 2

# Definir el tamaño del recorte deseado (un cuadrado de lado X lado)
lado_recorte = 300  
radio = lado_recorte // 2

# Calcular las coordenadas de inicio (arriba/izquierda) y fin (abajo/derecha)
inicio_y = int(centro_y - radio)
fin_y    = int(centro_y + radio)
inicio_x = int(centro_x - radio)
fin_x    = int(centro_x + radio)

# Las imágenes en OpenCV son matrices de NumPy.
# El recorte se hace así: imagen[filas_inicio:filas_fin, columnas_inicio:columnas_fin]
# Es decir: imagen[Y_inicio:Y_fin, X_inicio:X_fin]
cropped_img = img[inicio_y:fin_y, inicio_x:fin_x]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Imagen 1: Original completa
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title(f'Original ({ancho}x{alto})')

# Imagen 2: Recorte central
if cropped_img.size > 0:
    ax2.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Recorte Central ({lado_recorte}x{lado_recorte})')
    ax2.axis('off')
else:
    print("Error: La imagen original es más pequeña que el tamaño de recorte deseado.")
    ax2.set_title('Error al recortar')

plt.tight_layout()
plt.show()
