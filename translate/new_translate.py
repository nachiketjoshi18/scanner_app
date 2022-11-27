from textblob import TextBlob
import pytesseract
import argparse
import cv2
import os 

def translate(img):
    image = cv2.imread(img)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    text = text.replace("\n", " ")
    tb = TextBlob(text)
    translated = tb.translate(from_lang='en', to='hi')
    return translated


directory = os.path.join("C:/Users/NACHIKET JOSHI/Desktop/OCR/images")
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            pre_fix=file[:-4]
            txt=str(translate("C:/Users/NACHIKET JOSHI/Desktop/OCR/images/"+file))
            with open(directory+"/"+pre_fix+".txt",'w',encoding="utf-8") as f: f.write(txt)
