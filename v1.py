
import numpy  as np
import cv2

#使用大津法进行二值化


img1=cv2.imread("./img/front/dark/d2.jpg")
# img1=cv2.imread("./img/front/pic30.jpg")
# img1=cv2.imread("./img/多角度拍摄/角度2/IMG_1795.jpg")

(B, G, R) = cv2.split(img1)
zeros = np.zeros(img1.shape[:2], dtype = "uint8")

# 分别扩展B、G、R成为三通道。另外两个通道用上面的值为0的数组填充
img1 = cv2.merge([zeros, G, R])
cv2.imwrite("./v1/GR.jpg", cv2.merge([zeros, G, R]))
# cv2.imwrite("./v1/G.jpg", cv2.merge([zeros, G, zeros]))
# cv2.imwrite("./v1/R.jpg", cv2.merge([zeros, zeros, R]))

sobelx = cv2.Sobel(img1,cv2.CV_64F, 1, 0, ksize=1)
sobelx = cv2.convertScaleAbs(sobelx)
cv2.imwrite('./v1/sobelx.jpg',sobelx)
canny = cv2.Canny(cv2.GaussianBlur(img1, (5, 5), 0), 50, 150)
cv2.imwrite('./v1/canny.jpg',canny)
# img1 = cv2.GaussianBlur(img1, (5, 5), 0)
imgGray = cv2.cvtColor(sobelx,cv2.COLOR_BGR2GRAY)
# imgGray = cv2.Laplacian(imgGray, cv2.CV_8U)

# imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
threshold,imgAdapt = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('./v1/imgOtsu.jpg',imgAdapt)

