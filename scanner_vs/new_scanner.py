from four_pt_transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import os

i=2

def scanner(img):
    image = cv2.imread(img)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    output = imutils.resize(warped, height = 650)

    return output

j =1
while j !=0:
    directory = os.path.join("C:/Users/NACHIKET JOSHI/Desktop/scanner_vs/image")
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                pre_fix=file[:-4]
                img=scanner("C:/Users/NACHIKET JOSHI/Desktop/scanner_vs/image/"+file)
                result = cv2.imwrite("C:/Users/NACHIKET JOSHI/Desktop/scanner_vs/image"+pre_fix + str(i) +".jpg" , img)
                i=i+1
                if result:
                    print('Image is successfully saved as file.')
                j=0

                






    




