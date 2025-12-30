#histogramacolores.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
# Separar en canales
chans = cv2.split(img)

colors = ("b", "g", "r")
plt.figure()
plt.title("Histograma de Color")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
# Crear un histograma por cada canal
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()
