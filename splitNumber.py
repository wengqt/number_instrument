import cv2
import numpy as np

def processImg(dir):
    iii=11
    print('正在处理第 '+str(iii))
    # img1=cv2.imread("./img/front/dark/d"+str(iii)+".jpg")
    # img1=cv2.imread("./img/front/pic"+str(iii)+".jpg")
    # img1=cv2.imread("./img/多角度拍摄/角度1/a1_"+str(iii)+".jpg")
    img1=cv2.imread(dir)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(60, 60))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(40, 40))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6))


    imgBlur = cv2.GaussianBlur(img1, (5, 5), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./getArea4/gray.jpg',imgGray)




    # print(img1.shape[:2])



    sobelx = cv2.Sobel(imgGray,cv2.CV_64F, 1, 0, ksize=1)
    # sobely = cv2.Sobel(imgGray,cv2.CV_64F, 0, 1, ksize=3)
    # sobelxy = cv2.Sobel(imgGray,cv2.CV_64F, 1, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    cv2.imwrite('./getArea4/sobelx.jpg',sobelx)
    # cv2.imwrite('./getArea1/sobely.jpg',sobely)
    # cv2.imwrite('./getArea1/sobelxy.jpg',sobelxy)
    #图片二值化（自适应阈值）
    imgAdapt = cv2.adaptiveThreshold(sobelx,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,2)
    cv2.imwrite('./getArea4/adapt.jpg',imgAdapt)

    opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN, kernel4)
    cv2.imwrite('./getArea4/opened.jpg',opened)
    # opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN,kernel3)
    imgAdapt = cv2.erode(opened, kernel3, iterations=2)
    cv2.imwrite('./getArea4/erode.jpg',imgAdapt)
    imgAdapt =cv2.dilate(imgAdapt,kernel1,iterations=5)
    cv2.imwrite('./getArea4/opened1.jpg',imgAdapt)

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
    cv2.imwrite('./getArea4/opened_contours.jpg',cont_img)



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
    cv2.imwrite('./getArea4/cut.jpg',cutImg)
    # cutImg = cv2.adaptiveThreshold(cutImg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
    canny = cv2.Canny(cv2.GaussianBlur(cutImg, (5, 5), 0), 50, 150)
    cv2.imwrite('./getArea4/canny.jpg',canny)
    canny =cv2.dilate(canny,kernel3,iterations=1)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel2)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel3)
    cv2.imwrite('./getArea4/cannyDilate.jpg',canny)

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
    cv2.imwrite('./getArea4/line.jpg',img2)

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


    correct=cv2.warpAffine(canny,M,(width,hight))

    cv2.imwrite('./getArea4/correct.jpg',correct)


    ##分行
    (h1,w1)=correct.shape
    horizon = [0 for z in range(0, h1)]

    # canny1 = cv2.Canny(cutImg, 50, 150)
    # cv2.imwrite('./getArea4/correct_canny.jpg',canny1)
    # canny1 =cv2.dilate(canny1,kernel3,iterations=1)
    # cv2.imwrite('./getArea4/correct_cannyDilate.jpg',canny1)

    for i in range(0,h1): #遍历一行
        for j in range(0,w1):  #遍历一列
            if correct[i,j]==255:
                horizon[i]+=1

    newHorizon = np.zeros([h1, w1], np.uint8)


    for i in range(0,h1):
        for j in range(0,horizon[i]):
            newHorizon[i,j]=255

    cv2.imwrite('./getArea4/newHorizon.jpg',newHorizon)

    num1_min=0
    num1_max=0
    num2_min=0
    num2_max=0
    line_border = []


    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i - 20 if i - 20 > 0 else 0
                line_border.append(t)
                getLineZone2(src, i, n2)
                break

    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i + 20 if i + 20 <= n2 else n2
                line_border.append(t)
                getLineZone1(src, i, n2)
                break
            elif i>=n2:
                line_border.append(n2)

    getLineZone1(horizon,0,h1)

    # for i in range(0,h1):
    #    if horizon[i] !=0:
    #        num1_min = i-2
    #        break
    #
    # for i in range(num1_min,h1):
    #     if horizon[i] == 0 and horizon[i+1]==0:
    #         num1_max = i+2
    #         break
    #
    # for i in range(num1_max,h1):
    #     if horizon[i] != 0:
    #         num2_min = i-2
    #         break
    #
    # for i in range(num2_min,h1):
    #     if horizon[i] == 0 and horizon[i+1]==0:
    #         num2_max = i+2
    #         break

    print('line_border',line_border)
    # num1Img = correct[num1_min:num1_max, :]
    # num2Img = correct[num2_min:num2_max, :]
    #
    # cv2.imwrite('./getArea4/num1Img.jpg',num1Img)
    # cv2.imwrite('./getArea4/num2Img.jpg',num2Img)

    numLineImgs=[]



    #处理num1Img
    def splitNum(num_Img,index):
        cv2.imwrite('./getArea4/split'+str(int(index))+'Img.jpg', num_Img)
        num1closed = cv2.morphologyEx(num_Img, cv2.MORPH_CLOSE, kernel2)
        cv2.imwrite('./getArea4/num1closed.jpg',num1closed)

        (hI1,wI1)=num1closed.shape
        vertical1 = [0 for z in range(0, wI1)]

        for i in range(0,wI1): #遍历一lie
            for j in range(0,hI1):  #遍历一hang
                if num1closed[j,i]==255:
                    vertical1[i]+=1

        num1Vertical = np.zeros([hI1, wI1], np.uint8)


        for i in range(0,wI1):
            for j in range(0,vertical1[i]):
                num1Vertical[j,i]=255

        cv2.imwrite('./getArea4/num1Vertical.jpg',num1Vertical)

        num1points=[]

        #src 为一维数组
        def getSingleZone1(src,n1,n2):
            for i in range(n1,n2):
                if src[i] !=0:
                    t = i - 20 if i - 20 > 0 else 0
                    num1points.append(t)
                    getSingleZone2(src,i,n2)
                    break

        def getSingleZone2(src,n1,n2):
            for i in range(n1,n2):
                if src[i] ==0:
                    t = i + 20 if i + 20 <= n2 else n2
                    num1points.append(t)
                    getSingleZone1(src,i,n2)
                    break
                elif i >= n2:
                    num1points.append(n2)

        getSingleZone1(vertical1,0,wI1)


        print('num1points',num1points)
        maxW=0
        for i in range(0,len(num1points),2):
           maxW =  num1points[i+1]-num1points[i] if maxW<num1points[i+1]-num1points[i] else maxW



        allImgs = []
        for i in range(0,len(num1points),2):
            num1points[i] = num1points[i + 1] - maxW
            cv2.imwrite('./getArea4/split'+str(index)+'_'+str(int(i/2))+'.jpg', num_Img[:,num1points[i+1]-maxW:num1points[i+1]])
            allImgs.append(num_Img[:,num1points[i]:num1points[i+1]])
        return allImgs

    # num1s = splitNum(num1Img,1)
    # num2s = splitNum(num2Img,2)

    nums =[]
    for i in range(0,len(line_border),2):
        newLine = correct[line_border[i]:line_border[i+1], :]
        numLineImgs.append(newLine)
        nums.append(splitNum(newLine,int(i/2)))
    ##处理小数点
    dots=[]
    for j in range(len(nums)):
        one_index=0
        dot=-1
        for oneimg in nums[j]:
            one_index+=1
            dot_line = cv2.HoughLines(oneimg, 1, np.pi / 180, 180)
            l_num=0
            ang=0
            for ls in dot_line:
                for line in ls:
                    rho = line[0]
                    theta = line[1]
                    if abs(theta) < np.pi / 6:
                        print('this angle : ', theta * 180 / np.pi)
                        l_num = l_num + 1
                        ang = ang + theta
            avg_angle = ang/float(l_num)*180/np.pi
            print('竖直夹角： ',avg_angle)
            oneimg_cp = oneimg.copy()
            oneimg_cp = cv2.dilate(oneimg_cp, kernel3, iterations=2)
            one_h,one_w = oneimg_cp.shape[:2]
            M_1 = cv2.getRotationMatrix2D((one_w / 2, one_h / 2), avg_angle , 1.0)

            oneimg_cp = cv2.warpAffine(oneimg_cp, M_1, (one_w, one_h))

            cv2.imwrite('./getArea4/correct'+str(j)+'_'+str(one_index)+'.jpg', oneimg_cp)
            (one_h1, one_w1) = oneimg_cp.shape
            # print((one_h1, one_w1))
            rr =[0 for z in range(0, one_w1)]
            for col in range(one_w1):
                for row in range(one_h1):
                    if oneimg_cp[row,col]==255:
                        rr[col]+=1
            p=[]
            for n in range(len(rr)):
                if rr[n] !=0:
                    p.append(n)
                    break
            if len(p)==1:
                for n in range(p[0],len(rr)):
                    if rr[n] ==0:
                        p.append(n)
                        break

            if len(p)==2:
                for n in range(p[1],len(rr)):
                    if rr[n] !=0:#说明有第二个峰值
                        p.append(n)
                        dot = one_index
                        break
        dots.append(dot)







    imgList = []
    for j in range(len(nums)):
        one_index=0
        arr=[]
        for oneimg in nums[j]:
            one_index+=1
            h_1, w_1 = oneimg.shape[:2]
            mask = np.zeros([h_1 + 2, w_1 + 2], np.uint8)
            cv2.floodFill(oneimg, mask,(0,0),(255,255,255), cv2.FLOODFILL_FIXED_RANGE)
            oneimg = cv2.bitwise_not(oneimg)
            oneimg= cv2.resize(oneimg,(128,256))
            cv2.imwrite('./test/'+str(iii)+'_'+str(j)+'_'+str(one_index)+'.jpg',oneimg)
            arr.append('./test/'+str(iii)+'_'+str(j)+'_'+str(one_index)+'.jpg')
        imgList.append(arr)
    return imgList,dots