import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import cv2
import numpy as np
import tensorflow as tf
import sys
print("Cargando modelo...")
try:
    model = tf.keras.models.load_model('mi_red_mnist.keras')
    print(" Modelo cargado correctamente.")
except Exception as e:
    print(f" Error cargando el modelo: {e}")
    sys.exit(1)
# Iniciar Webcam con gestión de errores
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: No se puede acceder a la webcam.")
    sys.exit(1)
cv2.namedWindow("Webcam en tiempo real", cv2.WINDOW_NORMAL)
cv2.namedWindow("Ojo de la IA", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ojo de la IA", 200, 200) 
print("Cámara iniciada. Pulsa 'q' en la ventana para salir-")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame (¿cámara desconectada?)")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        # Definir ROI 
        height, width = gray.shape
        # Hacemos el cuadro un poco más grande 
        tamano_caja = 200 
        x1 = int(width/2 - tamano_caja/2)
        y1 = int(height/2 - tamano_caja/2)
        x2 = int(width/2 + tamano_caja/2)
        y2 = int(height/2 + tamano_caja/2)
        # Recorte seguro (evita salirnos de la imagen)
        roi = gray[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if roi.size > 0: 
            roi_resized = cv2.resize(roi, (28, 28))
            roi_resized = cv2.bitwise_not(roi_resized)             
            roi_norm = roi_resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
            # Predicción
            prediccion = model.predict(roi_norm, verbose=0)
            clase = np.argmax(prediccion)
            confianza = np.max(prediccion)
            color = (0, 255, 0) if confianza > 0.7 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            texto = f"Num: {clase} ({confianza*100:.0f}%)"
            cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Mostrar lo que ve la IA en la ventana pequeña
            cv2.imshow("Ojo de la IA", roi_resized)
        # Mostrar ventana principal
        cv2.imshow("Webcam en tiempo real", frame)
        # Salida controlada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Salir.")
            break
except KeyboardInterrupt:
    print("Interrupción de teclado detectada.")
finally:
    print("Liberando recursos")
    cap.release()
    cv2.destroyAllWindows()
