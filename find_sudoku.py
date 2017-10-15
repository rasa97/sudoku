import cv2
import numpy as np


# img = cv2.imread('full.jpg', cv2.IMREAD_GRAYSCALE)
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img =  cv2.imread('full.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

cv2.imwrite('threshold2.jpg', thresh)

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

x,y,w,h = cv2.boundingRect(cnt)
res = img[y:y+h, x:x+w]
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
value = (img.shape[0] + img.shape[1])/2
resized_image = cv2.resize(thresh, (value, value))
cv2.imwrite("resize.jpg", resized_image)




# cv2.circle(img,(x,y), 5, (0,0,255), -1)
# cv2.circle(img,(x+w,y), 5, (0,0,255), -1)
# cv2.circle(img,(x,y+h), 5, (0,0,255), -1)
# cv2.circle(img,(x+w,y+h), 5, (0,0,255), -1)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
