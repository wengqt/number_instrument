import cv2
import numpy as np


kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# for iii in range(2,21):
#     print('正在处理第 '+str(iii))
#     # img1=cv2.imread("./img/front/dark/d"+str(iii)+".jpg")
#     # img1=cv2.imread("./img/front/pic"+str(iii)+".jpg")
#     img1=cv2.imread("./img/多角度拍摄/角度1/a1_"+str(iii)+".jpg")
#     # img1=cv2.imread(dir)
#
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(60, 60))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(40, 40))
#     kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#     kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6))
#
#
#     imgBlur = cv2.GaussianBlur(img1, (5, 5), 0)
#     imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('./get_train_area/gray.jpg',imgGray)
#
#
#
#
#     # print(img1.shape[:2])
#
#
#
#     sobelx = cv2.Sobel(imgGray,cv2.CV_64F, 1, 0, ksize=1)
#     # sobely = cv2.Sobel(imgGray,cv2.CV_64F, 0, 1, ksize=3)
#     # sobelxy = cv2.Sobel(imgGray,cv2.CV_64F, 1, 1, ksize=3)
#     sobelx = cv2.convertScaleAbs(sobelx)
#     cv2.imwrite('./get_train_area/sobelx.jpg',sobelx)
#     # cv2.imwrite('./getArea1/sobely.jpg',sobely)
#     # cv2.imwrite('./getArea1/sobelxy.jpg',sobelxy)
#     #图片二值化（自适应阈值）
#     imgAdapt = cv2.adaptiveThreshold(sobelx,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,2)
#     cv2.imwrite('./get_train_area/adapt.jpg',imgAdapt)
#
#     opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN, kernel4)
#     cv2.imwrite('./get_train_area/opened.jpg',opened)
#     # opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN,kernel3)
#     imgAdapt = cv2.erode(opened, kernel3, iterations=2)
#     cv2.imwrite('./get_train_area/erode.jpg',imgAdapt)
#     imgAdapt =cv2.dilate(imgAdapt,kernel1,iterations=5)
#     cv2.imwrite('./get_train_area/opened1.jpg',imgAdapt)
#
#     # cv2.imwrite('./getArea1/closed1.jpg',closed)
#
#     # 找到轮廓
#     _, contours, hierarchy = cv2.findContours(imgAdapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 绘制轮廓
#
#     c = sorted(contours, key=cv2.contourArea, reverse=True)
#     cont_img = img1.copy()
#     for i in range(len(c)):
#         rect = cv2.minAreaRect(c[i])
#         box = np.int0(cv2.boxPoints(rect))
#         cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
#
#     # 绘制结果
#     cv2.imwrite('./get_train_area/opened_contours.jpg',cont_img)
#
#
#
#     #
#     # #
#     # ##
#     # #上面采用的是diffpic1中的方法
#     # #
#     #
#     numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(c[0])))
#
#     Xs = [i[0] for i in numBox]
#     Ys = [i[1] for i in numBox]
#     x1 = min(Xs)
#     x2 = max(Xs)
#     y1 = min(Ys)
#     y2 = max(Ys)
#     hight = y2 - y1
#     width = x2 - x1
#     cutImg1 = imgBlur[y1:y1+int(hight/2), x1:x1+width]
#     cutImg2 = imgBlur[y1+int(hight/2):y1+hight, x1:x1+width]
#     cutImg1 = cv2.resize(cutImg1,(128,64))
#     cutImg2 = cv2.resize(cutImg2,(128,64))
#
#     cv2.imwrite('./get_train_area/cut'+str(iii)+'_1.jpg',cutImg1)
#     cv2.imwrite('./get_train_area/cut'+str(iii)+'_2.jpg',cutImg2)

def cutImage(findContoursImg, dst, origin,index=None):
    _, contours, hierarchy = cv2.findContours(findContoursImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    cont_img = origin.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    # 绘制结果
    cv2.imwrite('./get_train_area/red_contours.jpg', cont_img)

    numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(c[0])))
    Xs = [i[0] for i in numBox]
    Ys = [i[1] for i in numBox]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cut = origin[y1:y1 + hight, x1:x1 + width]
    if index is not None:
        cutImg1 = origin[y1:y1 + int(hight / 2)+20, x1:x1 + width]
        cutImg2 = origin[y1+int(hight/2)+20:y1+hight, x1:x1+width]
        cutImg1 = cv2.resize(cutImg1,(128,64))
        cutImg2 = cv2.resize(cutImg2,(128,64))

        cv2.imwrite('./get_train_area/cut'+str(index)+'_1.jpg',cutImg1)
        cv2.imwrite('./get_train_area/cut'+str(index)+'_2.jpg',cutImg2)
    cv2.imwrite('./get_train_area/' + dst, cut)
    return cut







def processImg(iii):
    # BFmatch(暴力匹配)：计算匹配图层的一个特征描述子与待匹配图层的所有特征描述子的距离返回最近距离。
    # 上代码：
    query = cv2.imread('./img/front/pic3.jpg')

    query = cv2.cvtColor(cv2.GaussianBlur(query, (5, 5), 0), cv2.COLOR_RGB2GRAY)

    # train = cv2.imread(dir)
    # train=cv2.imread('./img/多角度拍摄/角度4/a'+str(iii)+'.jpg')
    # train=cv2.imread('./img/cam2.png')
    # train=cv2.imread('./img/多角度拍摄/角度1/a1_5.jpg')
    train=cv2.imread('./img/多角度拍摄/角度2/IMG_1805.jpg')
    # train=cv2.imread('./img/front/pic5.jpg')
    # train=cv2.imread("./img/front/dark/d7.jpg")


    # 暴力匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(train, None)
    kp2, des2 = orb.detectAndCompute(query, None)
    # 针对ORB算法 NORM_HAMMING 计算特征距离 True判断交叉验证
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 特征描述子匹配
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    print('kp1: ', kp1)
    print('kp2: ', kp2)
    print(len(matches))
    img3 = cv2.drawMatches(train, kp1, query, kp2, matches[:100], None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite('./get_train_area/match.jpg', img3)

    im_zeros = np.zeros(query.shape, np.uint8)
    for point in kp1:
        cv2.circle(im_zeros, (int(point.pt[0]), int(point.pt[1])), 10, (255, 255, 255), -1)

    im_zeros = cv2.dilate(im_zeros, kernel2, iterations=5 )
    cv2.imwrite('./get_train_area/points.jpg', im_zeros)

    instrumentArea = cutImage(im_zeros, 'cut1.jpg', train)

    train = cv2.cvtColor(cv2.GaussianBlur(instrumentArea.copy(), (5, 5), 0), cv2.COLOR_RGB2GRAY)
    cutImg = train
    # 到目前位置，就识别出了仪器的位置
    # 接下来识别数字区域
    # sobelx = cv2.Sobel(cutImg, cv2.CV_64F, 1, 0, ksize=3)
    #
    # cutImg = cv2.convertScaleAbs(sobelx)
    cutImg = cv2.Canny(cutImg,20,300)
    # cutImg = cv2.Canny(cv2.GaussianBlur(cutImg, (5, 5), 0), 0, 150)
    cv2.imwrite('./get_train_area/sobelx.jpg', cutImg)
    threshold, cutImg = cv2.threshold(cutImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cutImg = cv2.adaptiveThreshold(cutImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./get_train_area/cutadapt.jpg', cutImg)
    # cutImg = cv2.morphologyEx(cutImg, cv2.MORPH_OPEN, kernel4)
    # cv2.imwrite('./v2/cutopened.jpg', cutImg)
    # cutImg = cv2.erode(cutImg, kernel3, iterations=1)
    # cv2.imwrite('./v2/cuterode.jpg', cutImg)
    cutImg = cv2.dilate(cutImg, kernel2, iterations=2)
    cv2.imwrite('./get_train_area/cutdilate.jpg', cutImg)

    cutImg = cutImage(cutImg, 'numZone.jpg', instrumentArea,iii)


for kkk in range(0,1):
    processImg(kkk)
    print('处理完',kkk)