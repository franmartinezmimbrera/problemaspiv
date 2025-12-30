import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

img = cv2.imread('personas.png')
if img is None:
    print("Error: No se encuentra la imagen 'taller_alfareria.jpg'")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()
    
    frontales = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    perfiles_normal = profile_cas.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    
    gray_flipped = cv2.flip(gray, 1)
    perfiles_flipped = profile_cas.detectMultiScale(gray_flipped, 1.1, 4, minSize=(40, 40))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for (x, y, w, h) in perfiles_normal:
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 0, 255), 3)

    ancho_img = img.shape[1]
    for (x, y, w, h) in perfiles_flipped:
        x_real = ancho_img - x - w
        cv2.rectangle(img_out, (x_real, y), (x_real+w, y+h), (0, 255, 255), 3)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title('Detección Completa de Personas (Frontal + Perfiles)')
    plt.axis('off')
    plt.show()
