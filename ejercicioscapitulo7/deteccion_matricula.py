import cv2
import numpy as np
import matplotlib.pyplot as plt

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
img = cv2.imread('coches.jpg')
if img is None:
    print("Error: No se encuentra la imagen.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detección inicial
    # Subimos minNeighbors para ser más exigentes con el patrón de texto
    raw_plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12, minSize=(50, 20))
    img_res = img.copy()
    count = 0
    #  Filtrado por Aspect Ratio (Proporción)
    for (x, y, w, h) in raw_plates:
        aspect_ratio = w / float(h)
        
        # Una matrícula suele tener un ratio entre 2.5 y 5.5
        if 2.5 < aspect_ratio < 5.5:
            cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img_res, f'Placa {aspect_ratio:.1f}', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            count += 1

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.title(f'Detección Filtrada: {count} objetos encontrados')
    plt.axis('off')
    plt.show()
