import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import os

"1:Load Data"

'preprocess the image and cut it in (40,40)'
def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    return resized

def get_steering():
    record_path ='input'
    record_list = []
    steering = []
    input_record_path = './input/'
    "record list"
    for i in os.listdir(record_path):
        record_list.append(i)
    "getting the steering values from the records"
    for j in record_list:
        record = input_record_path + j
        with open(record, 'r') as f:
            'load json file'
            input = json.load(f)
            for key, value in input.items():
                if type(value) != str and key != 'milliseconds' and key != 'user/throttle':
                    steering.append(value)
    return steering

def get_images():
    image_names_path = 'data_1'
    img_path = './data_1/'
    image = []
    processed = []
    "images name list"
    for img in os.listdir(image_names_path):
        image.append(img)
    "getting the preprocessed images"
    for i in image:
        images = img_path + i
        'pyplot function for array values of the image'
        array = plt.imread(images)
        'preprocess the image resize it to (40,40) and changeing it s color to gray for less values to train'
        array_processed = preprocess(array)
        processed.append(array_processed)
    return processed
'print(get_images())'
'print(get_steering())'


'changeing the type of our images and steering values'
feature = np.array(get_images()).astype('float32')
label = np.array(get_steering()).astype('float32')

'print(len(feature))'
'print(len(label))'

'''throttle = np.array(throttle_list).astype('float32')'''
'saving into pickle files'
with open("features", "wb") as f:
    pickle.dump(feature, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(label, f, protocol=4)
'''with open("throttle", "wb") as f:
    pickle.dump(throttle_list, f, protocol=4)'''
