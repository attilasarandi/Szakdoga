import numpy as np
import cv2
from keras.models import load_model
import pickle

model = load_model('model_save.h5')

def keras_predict(model, image):
    img = np.reshape(image, (-1, 40, 40, 1))
    steering_angle = float(model.predict(img, batch_size=1))
    steering_angle = steering_angle * 100
    return steering_angle

steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0
'''with open("throttle", "rb") as f:
    throttle = np.array(pickle.load(f))'''
cap = cv2.VideoCapture('video.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    steering_angle = keras_predict(model, img)
    print(steering_angle)

    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    'setting our wheel rotation by calling the cv2.RotationMatrix function'
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()