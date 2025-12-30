#ajuste_scale_factor.py
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('personas.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# scaleFactor=1.05 es mas lento pero detecta rostros mas pequeños
faces = face_cascade.detectMultiScale(gray, 1.05, 5)
for (x,y,w,h) in faces: cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
