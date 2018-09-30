import cv2
import numpy as np

##
# #注意：不sobel的时候是17，17，10，膨胀3次
# #使用sobel的时候高斯是5，5，0膨胀4次
# img1=cv2.imread("./img/front/dark/IMG_1987.jpg")


kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 4))
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(60, 60))


img1=cv2.imread("./img/front/pic30.jpg")
imgBlur = cv2.GaussianBlur(img1,(5,5),0)
cv2.imwrite('./getArea2/imgblur.jpg',img1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
# imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
threshold,imgAdapt = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('./getArea2/imgAdapt.jpg',imgAdapt)




bg = cv2.imread('./img/front/pic31.jpg')
bgBlur = cv2.GaussianBlur(bg, (5, 5), 0)
bgGray = cv2.cvtColor(bgBlur, cv2.COLOR_BGR2GRAY)
# bgAdapt = cv2.adaptiveThreshold(bgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
bgthreshold,bgAdapt = cv2.threshold(bgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('./getArea2/bgAdapt.jpg',bgAdapt)


diff1 = cv2.subtract(imgAdapt,bgAdapt)

# diffGray = cv2.subtract(imgGray,bgGray)
# sobelx = cv2.Sobel(diffGray,cv2.CV_64F, 1, 0, ksize=1)
# sobely = cv2.Sobel(diffGray,cv2.CV_64F, 0, 1, ksize=1)
# sobelxy = cv2.Sobel(diffGray,cv2.CV_64F, 1, 1, ksize=1)
# cv2.imwrite('./getArea2/sobelx.jpg',sobelx)
# cv2.imwrite('./getArea2/sobely.jpg',sobely)
# cv2.imwrite('./getArea2/sobelxy.jpg',sobelxy)
# diff1 = cv2.adaptiveThreshold(cv2.convertScaleAbs(sobelx), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3)


# diff1 = cv2.subtract(bgAdapt,imgAdapt)
cv2.imwrite('./getArea2/diff.jpg',diff1)
# diff1 = cv2.morphologyEx(diff1, cv2.MORPH_CLOSE,kernel3)
diff1 = cv2.erode(diff1, None, iterations=2)
cv2.imwrite('./getArea2/erode.jpg',diff1)
diff1 =cv2.dilate(diff1,kernel1,iterations=4)
# cv2.imwrite('./getArea2/diff.jpg',diff)
cv2.imwrite('./getArea2/diff1.jpg',diff1)
# cv2.imwrite('./getArea2/diffGray.jpg',diffGray)

# 找到轮廓
_, contours, hierarchy = cv2.findContours(diff1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓

c = sorted(contours, key=cv2.contourArea, reverse=True)
cont_img = img1.copy()
for i in range(len(c)):
    rect = cv2.minAreaRect(c[i])
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)

# 绘制结果
cv2.imwrite('./getArea2/opened_contours.jpg',cont_img)
