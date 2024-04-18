from keras.preprocessing import image
import numpy as np
from keras.models import load_model

model = load_model('cnn7.h5')

image_path = ''
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)

print("Predicted probabilities:", predictions)

predicted_class_index = np.argmax(predictions)

confidence = predictions[0][predicted_class_index]

if confidence < 0.6: #Уровень может варьироваться в зависимости от результата тренировки нейронной сети
    print('Unrecognized')
else:
    print("Predicted class index:", predicted_class_index)
    print(confidence)
