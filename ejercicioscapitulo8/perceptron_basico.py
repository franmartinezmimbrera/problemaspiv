#perceptron_basico.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam  
# Puerta AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])
model = Sequential([
    Input(shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.1), 
              metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)
test_data = np.array([[1,1]])
prediccion = model.predict(test_data, verbose=0)
print(f"Predicción para [1,1]: {prediccion[0][0]:.4f} (¿Es True?: {prediccion > 0.5})")
