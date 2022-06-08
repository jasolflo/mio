from PIL import Image
import cv2
import numpy as np
import keras

model = keras.models.load_model('C:/Users/User/Desktop/GTI/3/3b/ProyectoR/Reconocimiento/modelo_v1_3.h5')

path = r"C:\Users\User\Desktop\GTI\3\3b\ProyectoR\Reconocimiento\test_pepino.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (250, 250))
img = np.expand_dims(img, axis=0)
prediction = model.predict_on_batch((img))
prediction_result = np.argmax(prediction[0])
print(prediction)
if prediction_result == 0:
    res = "Nada"
elif prediction_result == 1:
    res = "Tomate"
elif prediction_result == 2:
    res = "Pepino"

print(res)