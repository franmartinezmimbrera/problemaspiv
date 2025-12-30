import cv2

cap = cv2.VideoCapture(0)

ret, frame_anterior = cap.read()
if not ret:
    print("Error al iniciar cámara")
    exit()

# Pre-procesamiento inicial
gray_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
gray_anterior = cv2.GaussianBlur(gray_anterior, (21, 21), 0)


cv2.namedWindow("Camara de Seguridad", cv2.WINDOW_NORMAL)
cv2.namedWindow("Diferencia (Lo que ve la IA)", cv2.WINDOW_NORMAL)

cv2.moveWindow("Camara de Seguridad", 50, 50)
cv2.moveWindow("Diferencia (Lo que ve la IA)", 700, 50)


while True:
    # Leemos frame actual
    ret, frame_actual = cap.read()
    if not ret: break

    # Pre-procesamiento
    gray_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    gray_actual = cv2.GaussianBlur(gray_actual, (21, 21), 0)

    # Calcular diferencia (Resta de imágenes)
    resta = cv2.absdiff(gray_anterior, gray_actual)

    # Umbralizar (Limpiar sombras leves)
    _, umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)
    umbral = cv2.dilate(umbral, None, iterations=2)

    # Buscar contornos del movimiento
    contornos, _ = cv2.findContours(umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        if cv2.contourArea(c) < 5000: 
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_actual, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_actual, "MOVIMIENTO DETECTADO", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    cv2.imshow("Camara de Seguridad", frame_actual)
    cv2.imshow("Diferencia (Lo que ve la IA)", umbral)

    # Actualizar referencia
    gray_anterior = gray_actual.copy()

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
