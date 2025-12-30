#deteccion_cuerpo.py
import cv2
import matplotlib.pyplot as plt
body_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
img = cv2.imread('personas.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bodies = body_cas.detectMultiScale(gray, 1.1, 3)
for (x,y,w,h) in bodies: cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
