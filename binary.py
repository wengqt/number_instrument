import cv2


img1=cv2.imread("./img/front/dark/IMG_1987.jpg")
imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# imgLap = cv2.Laplacian(imgGray, cv2.CV_8U)
# 大jin法
# threshold,imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #自适应二值阈值化
imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
cv2.imshow('SURF_features',imgAdapt)
cv2.waitKey()
cv2.destroyAllWindows()
# # cv2.imwrite("./imgAdaptIMG_1987.jpg", imgAdapt)
# # cv2.imwrite("./imgOtsuIMG_1987.jpg", imgOtsu)


# #sift特征提取
cv2.imshow('original',imgGray)
s = cv2.xfeatures2d.SIFT_create()
keypoints = s.detect(imgGray)
print(keypoints)

for k in keypoints:
    cv2.circle(imgGray,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)

cv2.imshow('SURF_features',imgGray)
cv2.waitKey()
cv2.destroyAllWindows()