#deteccion_gatos.py
import cv2
import matplotlib.pyplot as plt
cat_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
img = cv2.imread('perrogato.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cats = cat_cas.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in cats: cv2.rectangle(img,(x,y),(x+w,y+h),(0,128,255),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
