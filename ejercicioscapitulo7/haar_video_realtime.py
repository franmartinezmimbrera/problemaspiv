# haar_video_realtime.py
import cv2

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  Iniciar la captura de video (0 suele ser la webcam integrada)
cap = cv2.VideoCapture(0)

# Comprobar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
else:
    print("Presiona 'q' para salir.")
    
    while True:
        # Capturar frame a frame
        ret, frame = cap.read()      
        # Si el frame no se capturó bien, saltar
        if not ret:
            break
        # Pre-procesamiento: Escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #  Detección de rostros
        # Ajustamos a 1.1 para que sea más sensible que 1.3
        faces = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar rectángulos sobre el frame original
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #  MOSTRAR VIDEO (Función nativa de OpenCV)
        cv2.imshow('Deteccion Realtime', frame)

        #  Condición de salida: presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 
