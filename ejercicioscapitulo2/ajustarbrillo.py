#ajustarbrillo.py
import cv2
import matplotlib.pyplot as plt
# Cargar la imagen
img = cv2.imread('imagen.jpg')

# Ajustar el brillo (beta) y contraste (alpha)
# Usando la fórmula: g(x) = alpha * f(x) + beta
alpha = 1.5
beta = 50
img_ajustada = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Creamos una figura con 1 fila y 2 columnas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Imagen 1: Original
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')

# Imagen 2: Ajustada
ax2.imshow(cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB))
ax2.set_title(f'Ajustada (alpha={alpha}, beta={beta})')
ax2.axis('off')

plt.tight_layout() 
plt.show()
