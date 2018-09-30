import cv2
import numpy as np


# img1=cv2.imread("./img/front/dark/IMG_1987.jpg")
img1=cv2.imread("./img/front/pic30.jpg")
# img1=cv2.imread("./img/多角度拍摄/角度4/IMG_1836.jpg")

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



#
# #
# ##
# #上面采用的是diffpic1中的方法
# #
#
numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(c[0])))

Xs = [i[0] for i in numBox]
Ys = [i[1] for i in numBox]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
cutImg = imgGray[y1:y1+hight, x1:x1+width]
cv2.imwrite('./getArea1/cut.jpg',cutImg)

canny = cv2.Canny(cutImg, 50, 200)
cv2.imwrite('./getArea1/canny.jpg',canny)
canny =cv2.dilate(canny,kernel3,iterations=1)
cv2.imwrite('./getArea1/cannyDilate.jpg',canny)

lines = cv2.HoughLines(canny,1,np.pi/180,180)
# lines1 = cv2.HoughLinesP(canny,1,np.pi/180,60)
# print(lines)
numLine=0
angle = 0
#
# for x11,y11,x22,y22 in lines1[0]:
# 	cv2.line(cutImg,(x11,y11),(x22,y22),(0,255,0),2)
#
# cv2.imwrite('./getArea1/line.jpg',cutImg)
lines1 = lines[:,0,:]
img2 = img1[y1:y1+hight, x1:x1+width]
for rho,theta in lines1[:]:
    if abs(np.pi / 2 - theta) < np.pi / 6 :
        print('a angle is :',theta*180/np.pi,'and ',(np.pi / 2 - theta)*180/np.pi)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x11 = int(x0 + 1000*(-b))
        y11 = int(y0 + 1000*(a))
        x22 = int(x0 - 1000*(-b))
        y22 = int(y0 - 1000*(a))
        cv2.line(img2,(x11,y11),(x22,y22),(0,0,255),3)
cv2.imwrite('./getArea1/line.jpg',img2)

for ls in lines:
    for line in ls:
        rho = line[0]
        theta = line[1]

        # if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        #     print('Vertical line , rho : %f , theta : %f' % (rho, theta))
        #     pt1 = (int(rho / np.cos(theta)), 0)
        #     # pt2 = (int((rho - magMat.shape[0] * np.sin(theta)) / np.cos(theta)), magMat.shape[0])
        #     angle = theta
        # else:
        #     print('Horiz line , rho : %f , theta : %f' % (rho, theta))
        #     pt1 = (0, int(rho / np.sin(theta)))
        #     # pt2 = (magMat.shape[1], int((rho - magMat.shape[1] * np.cos(theta)) / np.sin(theta)))
        #     # cv.line(lineImg, pt1, pt2, (255), 1)
        #     angle = theta + np.pi / 2
        if abs(np.pi/2 - theta) < np.pi / 6:
            print('this angle : ',theta*180/np.pi)
            numLine = numLine+1
            angle = angle + theta


        # if angle > (np.pi / 2):
        #     angle = angle - np.pi

    print('angle : %f, split : %f' % (angle,numLine))

averageAngle = (angle/float(numLine))*180/np.pi
print('averageAngle : %f' % averageAngle)

M=cv2.getRotationMatrix2D((width/2,hight/2),averageAngle-90,1.0)


dst=cv2.warpAffine(canny,M,(width,hight))

cv2.imwrite('./getArea1/correct.jpg',dst)
