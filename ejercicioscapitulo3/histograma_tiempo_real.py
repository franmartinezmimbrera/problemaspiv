import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error cámara")
    exit()

cv2.namedWindow("Cámara", cv2.WINDOW_NORMAL)

start = time.time()
hist_acum = np.zeros((256, 1), dtype=np.float32)

# CAPTURA 10 SEGUNDOS
while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_acum += hist

    cv2.imshow("Cámara", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Cámara")  


hist_acum = cv2.normalize(hist_acum, None, 0, 300, cv2.NORM_MINMAX)
canvas = np.zeros((300, 256), dtype="uint8")

for i in range(1, 256):
    cv2.line(
        canvas,
        (i - 1, 300 - int(hist_acum[i - 1][0])),
        (i,     300 - int(hist_acum[i][0])),
        255, 1
    )

cv2.namedWindow("Histograma final (10s)", cv2.WINDOW_NORMAL)
cv2.imshow("Histograma final (10s)", canvas)

cv2.waitKey(0)            
cv2.destroyAllWindows()

# Mostrar histograma final en el notebook
plt.figure(figsize=(8, 4))
plt.imshow(canvas, cmap="gray")
plt.title("Histograma final (10 segundos)")
plt.axis("off")
plt.show()
