import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/mnist_cnn.h5")

def predict_digit(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28)
    prediction = model.predict(img)
    return np.argmax(prediction)
