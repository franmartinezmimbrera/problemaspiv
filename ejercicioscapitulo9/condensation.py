#condensation.py
import numpy as np
import cv2

# Configuración
n_particles = 200
width, height = 800, 600
particles = np.random.rand(n_particles, 2) * [width, height] # Partículas aleatorias
# Mouse actúa como el objeto real a rastrear
target = np.array([width/2, height/2])
def mouse(event, x, y, flags, param):
    global target
    target = np.array([x, y])

cv2.namedWindow('Particle Filter')
cv2.setMouseCallback('Particle Filter', mouse)
while True:
    frame = np.zeros((height, width, 3), dtype=np.uint8)    
    #  PREDICCIÓN (Mover partículas + Ruido)
    noise = np.random.randn(n_particles, 2) * 15
    particles += noise
    # EVALUACIÓN (Calcular pesos)
    # Las partículas cerca del mouse (target) pesan más
    dist = np.linalg.norm(particles - target, axis=1)
    # Función Gaussiana de similitud: más cerca = peso más alto (hasta 1.0)
    weights = np.exp(-dist**2 / (2 * 50**2))
    weights /= np.sum(weights) # Normalizar para que sumen 1
    
    #  REMUESTREO (Supervivencia del más apto)
    # Seleccionamos índices basados en su peso
    indices = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
    particles = particles[indices]    
    # Visualización
    for p in particles:
        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1) # Partículas azules
    
    cv2.circle(frame, (int(target[0]), int(target[1])), 10, (0, 0, 255), -1) # Objetivo Rojo    
    # Centro de masa estimado (la "respuesta" del filtro)
    estimated = np.mean(particles, axis=0)
    cv2.circle(frame, (int(estimated[0]), int(estimated[1])), 5, (0, 255, 255), 2) # Estimación Amarilla

    cv2.imshow('Particle Filter', frame)
    if cv2.waitKey(20) & 0xFF == 27: break

cv2.destroyAllWindows()
