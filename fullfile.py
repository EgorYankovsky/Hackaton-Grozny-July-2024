# Подключаем необходмые пакеты

from Modules.template import *
from Modules.model import *
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Объявляем рабочие директории

DATASET_PATH = 'images/'
MODEL_PATH = './resnet18_letters.pth'
LOCAL_PATH = 'temp_crops/'
images_pth = ['./images/' + p.name for p in Path('images').iterdir()]

# Функции для записи и удаления изображений

def writeTo(images, path):
    for i, image in enumerate(images):
        cv.imwrite(f'{path}{i}.jpg', image)

def removeFrom(path):
    for f in Path(path).iterdir():
        os.remove(f.absolute())

# Step 1

def image_segmentation(img):
    img = cv.resize(img, (512, 112), fx=1, fy=1, interpolation=cv.INTER_CUBIC)

    model = YOLO('best.pt')
    model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = model.predict(img, verbose=False)
    mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).astype(np.uint8) # получение маски сегмента
    mask_raw = cv.resize(mask_raw, (512, 112), fx=1, fy=1, interpolation=cv.INTER_CUBIC)
    mask_raw = cv.merge((mask_raw,mask_raw,mask_raw))

    mask = cv.cvtColor(mask_raw, cv.COLOR_BGR2GRAY)
    return mask

# Step 2

def remove_unnecessary_vertexes(vertexes, w, h):
    left_up =    [w, h]
    right_up =   [0, h]
    right_down = [0, 0]
    for vertex in vertexes:
        if left_up[0] ** 2 + left_up[1] ** 2 > vertex[0] ** 2 + vertex[1] ** 2:
            left_up = vertex
        if (w - right_up[0]) ** 2 + right_up[1] ** 2 > (w - vertex[0]) ** 2 + vertex[1] ** 2:
            right_up = vertex
        if (w - right_down[0]) ** 2 + (h - right_down[1]) ** 2 > (w - vertex[0]) ** 2 + (h - vertex[1]) ** 2:
            right_down = vertex
    return [left_up, right_up, right_down]

def adaptive(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (27,27), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)
    return thresh

def global_thresh(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (17,17), 0)
    ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def stof_blur(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blur
    blur = cv.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
    # divide
    divide = cv.divide(gray, blur, scale=255)
    # otsu threshold
    thresh = cv.threshold(divide, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    return morph


def find_vertex(thresh):
    h, w = thresh.shape[0], thresh.shape[1]
    
    conts, hier = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(conts, key=cv.contourArea, reverse=True)[:5]
    
    points = []
    for cont in cnts:
        perimeter = cv.arcLength(cont, True)
        approx = cv.approxPolyDP(cont, 0.02 * perimeter, True)
        hull = cv.convexHull(approx, returnPoints=True)
        if len(hull) in range(1, 11) and cv.contourArea(hull) / 512 / 112 > 0.3:
            for point in hull:
                p = point[0]
                points.append(list(p))
    vert = remove_unnecessary_vertexes(points, w ,h)
    return vert

# Step 3


def make_affine_transformations(img, vertexes):
    rows,cols,ch = img.shape
    pts1 = np.float32([vertexes[0], vertexes[1], vertexes[2]])
    pts2 = np.float32([[0, 0], [512, 0], [512, 112]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(img,M,(cols,rows))

    return dst

# Функция для деформации изображения

def img_deformation(img):
  # Шаг 1.
  mask = image_segmentation(img)
  # Шаг 2.
  vertexes = find_vertex(mask)

  mask[mask == 1] = 255
  img = cv.resize(img, (512, 112), fx=1, fy=1, interpolation=cv.INTER_CUBIC)
  segmented_image = cv.bitwise_and(img, img, mask=mask)

  # Шаг 3.
  affined = make_affine_transformations(segmented_image, vertexes)
  return affined


def main():
    test = cv.imread(sys.argv[1])
    test = cv.resize(test,(512, 112), fx=1, fy=1, interpolation=cv.INTER_CUBIC)
    
    trans_img = img_deformation(test)

    crops2 = apply_template(trans_img, 2)
    crops3 = apply_template(trans_img, 3)
    writeTo(crops2, 'temp_crops/')
    writeTo(crops3, 'temp_crops/')

    model = LettersPrediction()

    return model.predict_series(crops2), model.predict_series(crops3) 


if __name__ == "__main__":
    main()
