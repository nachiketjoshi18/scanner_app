from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input receipt image")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

receiptCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        receiptCnt = approx
        break

if receiptCnt is None:
    raise Exception(("Could not find receipt outline. " "Try debugging your edge detection and contour steps."))


if args["debug"] > 0:
    output = image.copy()
    cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Receipt Outline", output)
    cv2.waitKey(0)

receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

cv2.imshow("Receipt Transform", imutils.resize(receipt, width=500))
cv2.waitKey(0)

options = "--psm 4"
rgb = cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb)
text = text.replace("\n", " ")

print("[INFO] raw output:")
print("==================")
print(text)
print("\n")


