from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
# Usaremos PIL (necesario para metadatos EXIF) 
imagen_pil = Image.open(ruta_imagen)
print(f"Formato: {imagen_pil.format}")
print(f"Tamaño (ancho x alto): {imagen_pil.size}")
print(f"Modo de color: {imagen_pil.mode}")
exif_data = imagen_pil._getexif()
print("Información mediante librería PIL ")
if exif_data is not None:
    for etiqueta, valor in exif_data.items():
            nombre = ExifTags.TAGS.get(etiqueta, etiqueta)
            print(f"{nombre}: {valor}")
else:
    print("La imagen no contiene metadatos EXIF.")

imagen_cv = cv2.imread(ruta_imagen)
if imagen_cv is not None:
    alto, ancho, canales = imagen_cv.shape
    print("Información mediante librería OpenCV")
    print(f"Resolución: {ancho} x {alto} píxeles")
    print(f"Número de canales: {canales}")
    print(f"Tipo de datos: {imagen_cv.dtype}")
else:
    print("\nOpenCV no pudo cargar la imagen.")
plt.imshow(cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
