import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
print(" Cargando MobileNetV2...")
try:
    model = MobileNetV2(weights='imagenet')
    print(" Modelo cargado.")
except Exception as e:
    print(f" Error cargando modelo")
    exit()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" No se detecta la cámara.")
    exit()
print("Iniciando. Pulsa 'q' o cierra la ventana para salir")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Error de lectura de cámara.")
            break
        # MobileNet espera imágenes de 224x224
        img_resized = cv2.resize(frame, (224, 224))
        # Preprocesamiento
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)
        # Predicción
        preds = model.predict(img_preprocessed, verbose=0)
        resultados = decode_predictions(preds, top=3)[0]
        texto_top = f"{resultados[0][1]}: {resultados[0][2]*100:.1f}%"
        cv2.putText(frame, texto_top, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        texto_segundo = f"o {resultados[1][1]}: {resultados[1][2]*100:.1f}%"
        cv2.putText(frame, texto_segundo, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Detector de Objetos', frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        if cv2.getWindowProperty('Detector de Objetos', cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    print("\n Interrumpido por el usuario (Ctrl+C).")
finally:
    cap.release()
    cv2.destroyAllWindows()
