import cv2
import numpy as np
import random

# img1=cv2.imread("./img/front/dark/d2.jpg")
img1=cv2.imread("./img/front/pic30.jpg")
# img1=cv2.imread("./img/多角度拍摄/角度2/IMG_1795.jpg")

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(18, 18))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 4))
#图片灰度化

# imgBlur = cv2.GaussianBlur(img1,(7,7),0)
# cv2.imwrite('./getArea/imgblur.jpg',imgBlur)
imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# h,w = imgGray.shape[:2]
# for i in range(h):
#     for j in range(w):
#         # if i== h-10:
#             # print(imgGray[i,j])
#         if imgGray[i,j] >251:
#             imgGray[i,j] = imgGray[i,j]-random.random()*120

# print(imgGray)
# img2 = cv2.morphologyEx(imgGray,cv2.MORPH_BLACKHAT,kernel1)
# img2 = cv2.equalizeHist(img2)

cv2.imwrite('./getArea/gray.jpg',imgGray)

#图片二值化（自适应阈值）
imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
cv2.imwrite('./getArea/adapt.jpg',imgAdapt)



# print(img1.shape[:2])


closed = cv2.morphologyEx(imgAdapt, cv2.MORPH_CLOSE, kernel1)
opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN,kernel1)
cv2.imwrite('./getArea/opened1.jpg',opened)
# 形态学梯度
# gradient = cv2.morphologyEx(imgAdapt, cv2.MORPH_GRADIENT, kernel1)


# topHat1 = cv2.morphologyEx(imgAdapt,cv2.MORPH_TOPHAT,kernel1)
# blackHat = cv2.morphologyEx(imgAdapt,cv2.MORPH_BLACKHAT,kernel1)
# perform a series of erosions and dilations


#闭运算 可有可不有  感觉没有什么效果。
# opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel3)
# cv2.imwrite('./getArea/opened2.jpg',opened)

opened = cv2.erode(opened, kernel3, iterations=4)
cv2.imwrite('./getArea/opened3.jpg',opened)
opened = cv2.dilate(opened, kernel2, iterations=5)
cv2.imwrite('./getArea/opened4.jpg',opened)


# 找到轮廓
_, contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓

c = sorted(contours, key=cv2.contourArea, reverse=True)
cont_img = img1.copy()
for i in range(len(c)):
    rect = cv2.minAreaRect(c[0])
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)

# 绘制结果
cv2.imwrite('./getArea/opened_contours.jpg',cont_img)

# copyImage = imgAdapt.copy()#复制原图像
# h, w = imgAdapt.shape[:2]#读取图像的宽和高
# mask = np.ones([h+2, w+2], np.uint8)#新建图像矩阵  +2是官方函数要求
# # cv2.floodFill(copyImage, mask, (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
# cv2.floodFill(copyImage, mask,(0, 80), (0, 100, 255))
# cv2.imwrite('./getArea/flood.jpg',copyImage)


# cv2.imshow('自适应阈值',imgAdapt)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imwrite('./getArea/topHat.jpg',topHat1)
# cv2.imwrite('./getArea/blackHat.jpg',blackHat)
cv2.imwrite('./getArea/opened.jpg',opened)
cv2.imwrite('./getArea/closed.jpg',closed)
# cv2.imwrite('./getArea/gradient.jpg',gradient)

