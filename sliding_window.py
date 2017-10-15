import cv2
import time
import numpy as np

img = cv2.imread('s1.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

wind_row, wind_col = img.shape[0]/9, img.shape[1]/9

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x+4, y+4, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_window():
    n=1
    for(x,y, window) in sliding_window(img, img.shape[0]/9, (wind_row,wind_col)):
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue
        clone = img.copy()
        cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        t_img = img[y:y+wind_row-1,x:x+wind_col-3]
        t_img = t_img[int(t_img.shape[1]*(0.08)):int(t_img.shape[1]*(0.95)), int(t_img.shape[0]*(0.08)):int(t_img.shape[0]*(0.93))]
        m='d/'+str(n)+'.jpg'
        cv2.imwrite(m,t_img)
        n=n+1
        cv2.waitKey(1)
show_window()
