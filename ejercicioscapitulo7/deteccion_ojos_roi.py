#deteccion_ojos_roi
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('personas.png')
if img is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Convertir a gris para el procesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()
    # Detectar Rostros
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        # Dibujar rectángulo en el rostro (Azul)
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Definir las Regiones de Interés (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_out[y:y+h, x:x+w]
        
        # RESTRICCIÓN PARA EVITAR LA BOCA 
        # Cortamos el ROI de la cara para que solo busque en el 60% superior
        alto_ojos = int(h * 0.6)
        zona_ojos_gray = roi_gray[0:alto_ojos, :]
        
        #  Detectar Ojos en la zona restringida
        # minNeighbors=10 es lo bastante estricto para evitar falsos positivos
        eyes = eye_cas.detectMultiScale(
            zona_ojos_gray, 
            scaleFactor=1.05, 
            minNeighbors=10, 
            minSize=(30, 30)
        )
        
        for (ex, ey, ew, eh) in eyes:
              cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

   
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title('Detección Fina: Rostros y Ojos (Sin Boca)')
    plt.axis('off')
    plt.show()
