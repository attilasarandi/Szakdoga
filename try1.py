import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle

img = cv2.imread("data_1/6176_cam-image_array_.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
plt.imshow(img)
plt.show()
'''
print(img.shape)
height = img.shape[0]
width = img.shape[1]

"print(height, width)"
'print(height/2)'
'print(width/2)'


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    "channel_count = img.shape[2]"
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

region_of_interest_vertices = [
    (0, width),
    (0, height/2),
    (width/3, height/10),
    (width, height/2),
    (width, height)
]

"szinesbol szurke kepet csinalunk"
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"ennek a szurke kepnek a vonalait keressuk"
canny = cv2.Canny(gray_image, 100, 200)
"levagjuk az img-bol a felesleges reszt"
"gauss = cv2.bilateralFilter(img,15,100,100)"

cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int64))

lines = cv2.HoughLinesP(cropped_image,
                        rho=5,
                        theta=np.pi/120,
                        threshold=110,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=15)

image_with_lines = drow_the_lines(img, lines)

plt.imshow(lines)
plt.show()