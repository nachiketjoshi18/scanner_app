import argparse
import cv2
import os

i=1
def togray(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

j=1
while j != 0:
    directory = os.path.join("C:/Users/NACHIKET JOSHI/Desktop/color change/image")
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                pre_fix=file[:-4]
                img=togray("C:/Users/NACHIKET JOSHI/Desktop/color change/image/"+file)
                result = cv2.imwrite("C:/Users/NACHIKET JOSHI/Desktop/color change/image"+pre_fix + str(i) +".jpg" , img)
                if result:
                    print('Image is successfully saved as file.')

