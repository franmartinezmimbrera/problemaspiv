#deteccion_torso.py
import cv2
import matplotlib.pyplot as plt
upper_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
img = cv2.imread('personas.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
uppers = upper_cas.detectMultiScale(gray, 1.1, 3)
for (x,y,w,h) in uppers: cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,255),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
