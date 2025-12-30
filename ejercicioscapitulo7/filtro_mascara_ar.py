#filtro_mascara_ar.py
import cv2
import matplotlib.pyplot as plt
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('imagen.jpg')
# Simulamos un rectangulo de gafas
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cas.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.circle(img, (x + w//4, y + h//3), 10, (0,0,0), -1)
    cv2.circle(img, (x + 3*w//4, y + h//3), 10, (0,0,0), -1)
    cv2.line(img, (x + w//4, y+h//3), (x + 3*w//4, y+h//3), (0,0,0), 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
