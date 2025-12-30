#filtro_kalman.py
import cv2
import numpy as np

# Lienzo negro
frame = np.zeros((600, 800, 3), np.uint8)

# CONFIGURACIÓN KALMAN 
# 4 variables de estado (x, y, dx, dy), 2 de medición (x, y)
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32) # Solo medimos posición
kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32) # Física básica
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Función para capturar mouse
last_measurement = current_measurement = np.array((2,1), np.float32)
def mousemove(event, x, y, flags, param):
    global current_measurement, last_measurement
    last_measurement = current_measurement
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

cv2.namedWindow("Kalman Tracker")
cv2.setMouseCallback("Kalman Tracker", mousemove)

while True:
    #  PREDECIR (El cuadrado VERDE es donde la IA cree que estarás)
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])
    
    #  CORREGIR (Usamos la posición real del mouse)
    kalman.correct(current_measurement)
    
    # Dibujar
    cv2.line(frame, (int(last_measurement[0]), int(last_measurement[1])), 
             (int(current_measurement[0]), int(current_measurement[1])), (0,255,0), 1) # Ruta real
    
    cv2.rectangle(frame, (pred_x-10, pred_y-10), (pred_x+10, pred_y+10), (0,0,255), 2) # Predicción (Rojo)
    
    cv2.imshow("Kalman Tracker", frame)
    if cv2.waitKey(30) & 0xFF == 27: break # ESC para salir

cv2.destroyAllWindows()
