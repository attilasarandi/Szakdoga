import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

"1:Load Data"

'preprocess the image and cut it in (40,40)'
def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    return resized

feature = []
features = './data_1/'
labels = []
label = []

'getting the data from individual json files'
with open('input/record_100.json', 'r') as f:
    file = json.load(f)
    for i in file.values():
        labels.append(i)

'deleting last 2 of the values'
labels.pop(3)
labels.pop(3)
'current label list which contains the angle and throttle'
'print(labels)'

image = labels[0]
'connecting the path to the image'
img_path = features + image
'reading the image'
img = plt.imread(img_path)
'preprocessing'
feature.append(preprocess(img))
'print(feature)'

'label file fro the training dataset'
label.append(labels[1])

feature = np.array(feature).astype('float32')
label = np.array(label).astype('float32')

print(len(feature))
print(len(label))

'saving into pickle files'
with open("features_40", "wb") as f:
    pickle.dump(feature, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(label, f, protocol=4)
