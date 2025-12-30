# deteccion_perfil.py
import cv2
import matplotlib.pyplot as plt

profile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
img = cv2.imread('personas.png')

if img is None:
    print("Error: No se encuentra la imagen.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ancho_img = img.shape[1]

    # Subimos minNeighbors a 12 o 15 para ser muy estrictos
    # Definimos minSize para ignorar detalles pequeños como una oreja sola
    params = {
        "scaleFactor": 1.1,
        "minNeighbors": 15, 
        "minSize": (100, 100) 
    }

    #  Detección Normal
    profiles_normal = profile_cas.detectMultiScale(gray, **params)
    
    #  Detección Espejo
    gray_flipped = cv2.flip(gray, 1)
    profiles_flipped = profile_cas.detectMultiScale(gray_flipped, **params)

    for (x, y, w, h) in profiles_normal:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3)
        
    for (x, y, w, h) in profiles_flipped:
        x_real = ancho_img - x - w
        cv2.rectangle(img, (x_real, y), (x_real+w, y+h), (0, 255, 255), 3)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detección Estricta de Perfiles')
    plt.axis('off')
    plt.show()
