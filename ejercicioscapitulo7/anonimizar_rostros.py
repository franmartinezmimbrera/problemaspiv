import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'personas.png'  
img = cv2.imread(img_path)
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

if img is None:
    print(f"Error: No se encuentra la imagen {img_path}")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # Detección Frontal
    rostros = list(face_cas.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)))
    #  Detección Perfil (Normal)
    perfiles = list(profile_cas.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)))
    #  Detección Perfil (Espejo)
    gray_flipped = cv2.flip(gray, 1)
    perfiles_flipped = list(profile_cas.detectMultiScale(gray_flipped, 1.1, 4, minSize=(30, 30)))
    # Ajustar coordenadas de los perfiles invertidos y añadirlos a la lista
    ancho_img = img.shape[1]
    for (x, y, w, h) in perfiles_flipped:
        x_real = ancho_img - x - w
        rostros.append((x_real, y, w, h))
    
    # Añadir los perfiles normales a la lista principal
    for rect in perfiles:
        rostros.append(rect)
    # ANONIMIZADO (Blur)
    img_anonima = img.copy()
    for (x, y, w, h) in rostros:
        # Extraer la región de interés (ROI) -> La cara
        roi_cara = img_anonima[y:y+h, x:x+w]       
        # Aplicar desenfoque Gaussiano fuerte
        # (99, 99) es el tamaño del kernel (debe ser impar)
        # 30 es la desviación estándar (cuanto más alto, más borroso)
        roi_blur = cv2.GaussianBlur(roi_cara, (99, 99), 30)        
        # Pegar la cara borrosa sobre la imagen original
        img_anonima[y:y+h, x:x+w] = roi_blur
        
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_anonima, cv2.COLOR_BGR2RGB))
    plt.title(f'Anonimizado completado: {len(rostros)} zonas censuradas')
    plt.axis('off')
    plt.show()
