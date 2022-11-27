from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re
import os



def ocr(file_to_ocr):
    print(file_to_ocr)
    orig = cv2.imread(file_to_ocr)
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

    receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

    options = "--psm 4"
    rgb = cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    text = text.replace("\n", " ") 

    return text



directory = os.path.join("C:/Users/NACHIKET JOSHI/Desktop/OCR/images")
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            pre_fix=file[:-4]
            txt=ocr("C:/Users/NACHIKET JOSHI/Desktop/OCR/images/"+file)
            with open(directory+"/"+pre_fix+".txt",'w') as f: f.write(str(txt))
    
    