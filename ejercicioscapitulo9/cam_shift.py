#cam_shift.py
import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Usa la webcam

# Tomar el primer frame
ret, frame = cap.read()
# Selecciona manualmente el objeto a rastrear (haz un recuadro con el mouse y pulsa ENTER)
r, h, c, w = cv2.selectROI("Selecciona Objeto", frame, False)
track_window = (c, r, w, h)

# Configurar la ROI para el rastreo
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Creamos una máscara para ignorar colores con poca luz (ruido)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # Usamos CamShift 
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
        # Dibujar una caja rotada que se adapta al tamaño
        pts = cv2.boxPoints(ret)
        pts = np.int64(pts)
        img2 = cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        
        cv2.imshow('Seguimiento CamShift', img2)
        if cv2.waitKey(1) & 0xFF == 27: break
    else: break

cap.release()
cv2.destroyAllWindows()
