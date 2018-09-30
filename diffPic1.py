import cv2
import numpy as np


# img1=cv2.imread("./img/front/dark/IMG_1987.jpg")
img1=cv2.imread("./img/front/pic30.jpg")
# img1=cv2.imread("./img/多角度拍摄/角度2/IMG_1795.jpg")

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(60, 60))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6))


imgBlur = cv2.GaussianBlur(img1, (5, 5), 0)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./getArea1/gray.jpg',imgGray)




# print(img1.shape[:2])



sobelx = cv2.Sobel(imgGray,cv2.CV_64F, 1, 0, ksize=1)
# sobely = cv2.Sobel(imgGray,cv2.CV_64F, 0, 1, ksize=3)
# sobelxy = cv2.Sobel(imgGray,cv2.CV_64F, 1, 1, ksize=3)
cv2.imwrite('./getArea1/sobelx.jpg',cv2.convertScaleAbs(sobelx))
# cv2.imwrite('./getArea1/sobely.jpg',sobely)
# cv2.imwrite('./getArea1/sobelxy.jpg',sobelxy)
#图片二值化（自适应阈值）
imgAdapt = cv2.adaptiveThreshold(cv2.convertScaleAbs(sobelx),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,2)
cv2.imwrite('./getArea1/adapt.jpg',imgAdapt)

opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN, kernel4)
cv2.imwrite('./getArea1/opened.jpg',opened)
# opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN,kernel3)
imgAdapt = cv2.erode(opened, kernel3, iterations=2)
cv2.imwrite('./getArea1/erode.jpg',imgAdapt)
imgAdapt =cv2.dilate(imgAdapt,kernel1,iterations=4)
cv2.imwrite('./getArea1/opened1.jpg',imgAdapt)

# cv2.imwrite('./getArea1/closed1.jpg',closed)

# 找到轮廓
_, contours, hierarchy = cv2.findContours(imgAdapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓

c = sorted(contours, key=cv2.contourArea, reverse=True)
cont_img = img1.copy()
for i in range(len(c)):
    rect = cv2.minAreaRect(c[i])
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)

# 绘制结果
cv2.imwrite('./getArea1/opened_contours.jpg',cont_img)

