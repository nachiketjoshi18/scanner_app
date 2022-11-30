import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str",
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
result = cv2.imwrite("C:\Users\NACHIKET JOSHI\Desktop\color change\results\result-1.jpg" , hsv)
if result:
    print('Image is successfully saved as file.')