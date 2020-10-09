import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('Autopilot.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    'steering_angle = steering_angle * 100'
    return steering_angle


def keras_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img
'min_speed = 10'
'max_speed = 30'
'speed_limit = max_speed'
smoothed_angle = 0
img = '100_cam-image_array_.jpg'
image = plt.imread(img)
gray = cv2.resize((cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
steering_angle = keras_predict(model, gray)


'if speed > speed_limit:'
'    speed_limit = min_speed'
'else:'
'    speed_limit = max_speed'
'if speed then -(speed/speed_limit)**2'

smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * \
                  (steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
throttle = 1.0 - smoothed_angle**2

print("Steering_angle:%.5f, throttle:%.5f" % (smoothed_angle*100, throttle*100))