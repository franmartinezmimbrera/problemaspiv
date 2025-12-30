#ajuste_min_neighbors.py
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('personas.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# minNeighbors=10 es mas estricto, evita falsos positivos
faces = face_cascade.detectMultiScale(gray, 1.3, 10)
for (x,y,w,h) in faces: cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
