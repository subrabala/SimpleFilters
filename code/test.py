import cv2 as cv
import numpy as np
from turtle import *
# ==============================DISPLAY=====================================

img=cv.imread('lena.jpg')
print(img) #prints image value (rgb) (500,500,3)
cv.imshow('LENA', img)  #1st parameter - name_of_window, 2nd - size


# ---------------------------RESIZE-----------------------------------------------------

h,w=img.shape[:2] #first 2 values in dimensions are h and w
new_h, new_w= int(h/2), int(w/2)
resize=cv.resize(img,(new_w, new_h))
cv.imshow('resizing', resize)
cv.imwrite('pic1.jpg',img)  #saves this image (2nd) named as (1st) in folder
cv.waitKey(3000) #waits for 3s before destroying all
cv.destroyAllWindows()

# ============================BLANK================================================

blank=np.zeros((500,500,3), dtype='uint8')
cv.imshow('BLANK', blank)
blank[0:100,100:300]=255,0,0  #height co-ord : width co-ord, 
cv.imshow('colour',blank)
cv.waitKey(10000)
pencolor('red')
fillcolor('yellow')

# ========================GRAYSCALE===================================

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

r,g,b= img[:,:,0], img[:,:,1], img[:,:,2]
cv.imshow('red', r)
cv.imwrite('gray.jpg',b)  #stores the grayscale image in folder

# ============================BLUR====================================

blur=cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)  #odd number tuple
cv.imshow('Blur',blur)
avg_blur=cv.blur(img,(5,5))
cv.imshow('Average Blur', avg_blur)
cv.waitKey(10000)

# ======================EDGE DETECTION=======================

edges=cv.Canny(img,175,120)
cv.imshow('edge-detect', edges)

# ==========================CROP=================================

crop_img = img[100:400, 200:600]
cv.imshow('Crop', crop_img)

# =======================COLOUR CHANNEL===========================

blank =np.zeros(img.shape[:2],dtype='uint8')
b,g,r =cv.split(img) #still black/white

blue= cv.merge([b,blank,blank])
cv.imshow('blue_channel', blue)  #only blue
green= cv.merge([blank,g,blank])
cv.imshow('green_channel', green)  #only green
red= cv.merge([blank,blank,r])
cv.imshow('red_channel', red)  #only red

merged= cv.merge([b,g,r])
cv.imshow('merge', merged)


# =========================CAT-FACE===================================

face_cascade=cv.CascadeClassifier("cat_face.xml")  

cat=cv.imread("cat3.jpg")
gray=cv.cvtColor(cat,cv.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
for x,y,w,h in faces:
    img1=cv.rectangle(cat,(x,y),(x+w,y+h),(57,255,68),3) #the image, starting co-ordinates, ending co-ordinates, colour of rectangle, width 
cv.imshow('Detected',img1)

#=========================NEGATIVE======================

img2=255-img
cv.imshow("negative", img2)
cv.imshow('original',img)

#=======================CIRCLE DETECTION====================
round=cv.imread("draw.png")
output=round.copy()
round_gray=cv.cvtColor(round,cv.COLOR_BGR2GRAY)
round_blur_gray=cv.blur(round_gray,(3,3))   #blur and gray filter
detected_circle = cv.HoughCircles(round_blur_gray, cv.HOUGH_GRADIENT,5,30)  #image, method (circle detection),,minimum distance
if detected_circle is not None:
    detected_circle=np.round(detected_circle[0,:]).astype("int")
    for (x,y,r) in detected_circle:
        cv.circle(output, (x,y),r,(255,0,0),3)
        cv.rectangle(output,(x-2,y-2),(x+2,y+2),(0,210,0),-1)
    cv.imshow("output",output)



cv.waitKey(2000)


