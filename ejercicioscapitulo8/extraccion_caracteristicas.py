import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Definir tamaño de imagen estándar para MobileNetV2
IMG_SHAPE = (224, 224, 3)

print(" Cargando base MobileNetV2")

#  Cargar el modelo BASE (Sin la cabeza de clasificación)
# include_top=False: Elimina la capa final que dice "Perro" o "Gato".
# weights='imagenet': Carga lo aprendido de millones de fotos.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

# Congelar la base 
# Esto evita que se destruya lo aprendido si fueras a entrenar algo encima.
base_model.trainable = False

# Crear el modelo extractor final
# Añadimos Pooling para convertir el mapa 7x7x1280 en un vector limpio de 1280 números
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

model.summary()

def obtener_embedding(img_path):
    # Cargar imagen y redimensionar a 224x224
    img = image.load_img(img_path, target_size=IMG_SHAPE[:2])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) # Crear lote de 1 imagen
    x = preprocess_input(x)       # Preprocesamiento específico de MobileNetV2

    # El modelo devuelve un vector de 1280 números
    feature_vector = model.predict(x)
    return feature_vector.flatten()

vector = obtener_embedding('personas.png')
print(f"Vector característico generado: {vector.shape}")

