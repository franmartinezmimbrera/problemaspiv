#deteccion_sonrisa.py
import cv2
import matplotlib.pyplot as plt
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
img = cv2.imread('personas.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cas.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    #Detectar sonrisas
    smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)
    for (sx,sy,sw,sh) in smiles: cv2.rectangle(img[y:y+h, x:x+w],(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
