import codecs
import os
from flask import *
from werkzeug.utils import secure_filename
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin
from fastai.vision.all import *
from PIL import Image
import numpy as np
import base64
from io import BytesIO, StringIO
import json
import logging
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import os
from textblob import TextBlob
import pytesseract
import re
import base64
import PIL.Image as pilimage
import io
import urllib.request
import requests
from io import BytesIO





#flask_api:-


app = Flask(__name__)
CORS(app)
api = Api(app)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# 1. Route for crop:-

@cross_origin()
@app.route('/crop', methods=['GET'])
def upload1():
    file = request.args.get('image')
    name = request.args.get('name')
    token = request.args.get('token')
    link = file[:78] + "%2F" + file[79:] + "&token=" + token
    resp = urllib.request.urlopen(link)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scan = scanner(image)

    res = pilimage.fromarray(scan)
    path = res.save(name)
    with open(name, 'rb') as ans:
        s = base64.b64encode(ans.read())

    return s

# 2. Route for OCR :-

@cross_origin()
@app.route('/ocr', methods=['GET'])
def upload2():
    file = request.args.get('image')
    token = request.args.get('token')
    link = file[:78] + "%2F" + file[79:] + "&token=" + token
    resp = urllib.request.urlopen(link)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scan = ocr(image)
    return scan

# 3. Route for translate:-

@cross_origin()
@app.route('/translate', methods=['GET'])
def upload3():
    file = request.args.get('image')
    lang = request.args.get('language')
    token = request.args.get('token')
    link = file[:78] + "%2F" + file[79:] + "&token=" + token
    resp = urllib.request.urlopen(link)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scan = translate(image,lang)
    return str(scan)

# 4 . Route for color change:-
@cross_origin()
@app.route('/color', methods=['GET'])
def upload4():
    file = request.args.get('image')
    name = request.args.get('name')
    token = request.args.get('token')
    link = file[:78] + "%2F" + file[79:] + "&token=" + token
    resp = urllib.request.urlopen(link)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scan = colorchange(image)

    res = pilimage.fromarray(scan)
    path = res.save(name)
    with open(name, 'rb') as ans:
        s = base64.b64encode(ans.read())

    return s

#automatic crop:-


def scanner(image):
    #image = cv2.imread(img)
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

# four_pt_transform:-

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


#translate

def translate(img,lang):
    language = lang
    image = img
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    text = text.replace("\n", " ")
    tb = TextBlob(text)
    translated = tb.translate(from_lang='en', to=language)
    return translated


#OCr:-

def ocr(file_to_ocr):
    rgb = cv2.cvtColor(file_to_ocr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    return text
   


# change color:-

def colorchange(img , color):
    colour = color
    if(colour == 'gray'):
        image = img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    if(colour == 'hsv'):
        image = img
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
    else:
        image = img
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

if __name__ == "__main__":
    app.run(debug=True)



