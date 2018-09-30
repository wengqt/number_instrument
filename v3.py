
import cv2
import numpy as np

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

##
# @param findContoursImg 膨胀之后用于找矩形轮廓的图
#        dst 目标生成的文件名
#        origin 用于切割的原图
def cutImage(findContoursImg, dst, origin):
    _, contours, hierarchy = cv2.findContours(findContoursImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    cont_img = origin.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    # 绘制结果
    cv2.imwrite('./v2/red_contours.jpg', cont_img)

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
    cv2.imwrite('./v2/' + dst, cut)
    return cut

def correctAngle(src):
    canny = cv2.Canny(cv2.GaussianBlur(src, (5, 5), 0), 0, 150)
    cv2.imwrite('./v2/numZonecanny.jpg', canny)
    canny = cv2.dilate(canny, kernel3, iterations=1)
    cv2.imwrite('./v2/numZonecannyDilate.jpg', canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 50)
    numLine = 0
    angle = 0

    (height,width) = src.shape[:2]

    for ls in lines:
        for line in ls:
            rho = line[0]
            theta = line[1]
            if abs(np.pi / 2 - theta) < np.pi / 6:
                # print('this angle : ', theta * 180 / np.pi)
                numLine = numLine + 1
                angle = angle + theta

            # if angle > (np.pi / 2):
            #     angle = angle - np.pi

    averageAngle = (angle / float(numLine)) * 180 / np.pi
    print('averageAngle : %f' % averageAngle)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), averageAngle - 90, 1.0)

    correct = cv2.warpAffine(canny, M, (width, height))

    cv2.imwrite('./v2/numZonecorrect.jpg', correct)
    return correct



def splitNum(num_Img, index):
    cv2.imwrite('./v2/split/split' + str(int(index)) + 'Img.jpg', num_Img)
    num1closed = cv2.morphologyEx(num_Img, cv2.MORPH_CLOSE, kernel5)
    cv2.imwrite('./v2/split/split' + str(int(index)) + 'closed.jpg', num1closed)

    (hI1, wI1) = num1closed.shape
    vertical1 = [0 for z in range(0, wI1)]

    for i in range(0, wI1):  # 遍历一lie
        for j in range(0, hI1):  # 遍历一hang
            if num1closed[j, i] == 255:
                vertical1[i] += 1

    # num1Vertical = np.zeros([hI1, wI1], np.uint8)
    # for i in range(0, wI1):
    #     for j in range(0, vertical1[i]):
    #         num1Vertical[j, i] = 255
    # cv2.imwrite('./v2/num1Vertical.jpg', num1Vertical)

    num1points = []

    # src 为一维数组
    def getSingleZone1(src, n1, n2):
        for i in range(n1, n2):
            if src[i] != 0:
                t = i - 10 if i - 10 > 0 else 0
                num1points.append(t)
                getSingleZone2(src, i, n2)
                break

    def getSingleZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i + 10 if i + 10 <= n2 else n2
                num1points.append(t)
                getSingleZone1(src, i, n2)
                break
            elif i>=n2-1:
                num1points.append(n2)

    getSingleZone1(vertical1, 0, wI1)

    print('num1points', num1points)
    maxW = 0
    for i in range(0, len(num1points), 2):
        maxW = num1points[i + 1] - num1points[i] if maxW < num1points[i + 1] - num1points[i] else maxW

    allImgs = []
    for i in range(0, len(num1points), 2):
        num1points[i] = num1points[i + 1] - maxW
        cv2.imwrite('./v2/split/split' + str(index) + '_' + str(int(i / 2)) + '.jpg',
                    num_Img[:, num1points[i + 1] - maxW:num1points[i + 1]])
        allImgs.append(num_Img[:, num1points[i]:num1points[i + 1]])
    return allImgs

def separateNum(correct_src,iii=11):
    ##分行
    (h1, w1) = correct_src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if correct_src[i, j] == 255:
                horizon[i] += 1

    # newHorizon = np.zeros([h1, w1], np.uint8)
    #
    # for i in range(0, h1):
    #     for j in range(0, horizon[i]):
    #         newHorizon[i, j] = 255
    #
    # cv2.imwrite('./getArea4/newHorizon.jpg', newHorizon)

    line_border = []

    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i-10 if i-10>0 else 0
                line_border.append(t)
                getLineZone2(src, i, n2)
                break

    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i+10 if i+10<=n2 else n2
                line_border.append(t)
                getLineZone1(src, i, n2)
                break
            elif i>=n2:
                line_border.append(n2)

    getLineZone1(horizon, 0, h1)
    print('line_border', line_border)
    numLineImgs = []
    nums = []
    for i in range(0, len(line_border), 2):
        newLine = correct_src[line_border[i]:line_border[i + 1], :]
        numLineImgs.append(newLine)
        nums.append(splitNum(newLine, int(i / 2)))
    ##处理小数点
    dots = []
    for j in range(len(nums)):
        one_index = 0
        dot = -1
        for oneimg in nums[j]:
            one_index += 1
            dot_line = cv2.HoughLines(oneimg, 1, np.pi / 180, 110)
            l_num = 0
            ang = 0
            for ls in dot_line:
                for line in ls:
                    rho = line[0]
                    theta = line[1]
                    if abs(theta) < np.pi / 6:
                        print('this angle : ', theta * 180 / np.pi)
                        l_num = l_num + 1
                        ang = ang + theta
            avg_angle = ang / float(l_num) * 180 / np.pi
            print('竖直夹角： ', avg_angle)
            oneimg_cp = oneimg.copy()
            oneimg_cp = cv2.dilate(oneimg_cp, kernel3, iterations=2)
            one_h, one_w = oneimg_cp.shape[:2]
            M_1 = cv2.getRotationMatrix2D((one_w / 2, one_h / 2), avg_angle, 1.0)

            oneimg_cp = cv2.warpAffine(oneimg_cp, M_1, (one_w, one_h))

            cv2.imwrite('./v2/split/correct' + str(j) + '_' + str(one_index) + '.jpg', oneimg_cp)
            (one_h1, one_w1) = oneimg_cp.shape
            # print((one_h1, one_w1))
            rr = [0 for z in range(0, one_w1)]
            for col in range(one_w1):
                for row in range(one_h1):
                    if oneimg_cp[row, col] == 255:
                        rr[col] += 1
            p = []
            for n in range(len(rr)):
                if rr[n] != 0:
                    p.append(n)
                    break
            if len(p) == 1:
                for n in range(p[0], len(rr)):
                    if rr[n] == 0:
                        p.append(n)
                        break

            if len(p) == 2:
                for n in range(p[1], len(rr)):
                    if rr[n] != 0:  # 说明有第二个峰值
                        p.append(n)
                        dot = one_index
                        break
        dots.append(dot)

    imgList = []
    for j in range(len(nums)):
        one_index = 0
        arr = []
        for oneimg in nums[j]:
            one_index += 1
            h_1, w_1 = oneimg.shape[:2]
            mask = np.zeros([h_1 + 2, w_1 + 2], np.uint8)
            cv2.floodFill(oneimg, mask, (0, 0), (255, 255, 255), cv2.FLOODFILL_FIXED_RANGE)
            oneimg = cv2.bitwise_not(oneimg)
            oneimg = cv2.resize(oneimg, (128, 256))
            cv2.imwrite('./v2/res/' + str(iii) + '_' + str(j) + '_' + str(one_index) + '.jpg', oneimg)
            arr.append('./v2/res/' + str(iii) + '_' + str(j) + '_' + str(one_index) + '.jpg')
        imgList.append(arr)
    print('dots:',dots)
    return imgList, dots




def processImg(dir):
    # BFmatch(暴力匹配)：计算匹配图层的一个特征描述子与待匹配图层的所有特征描述子的距离返回最近距离。
    # 上代码：
    iii = 11
    query = cv2.imread('./img/front/pic3.jpg')

    query = cv2.cvtColor(cv2.GaussianBlur(query, (5, 5), 0), cv2.COLOR_RGB2GRAY)

    # train = cv2.imread(dir)
    # train=cv2.imread('./img/多角度拍摄/角度4/a11.jpg')
    # train=cv2.imread('./img/cam2.png')
    # train=cv2.imread('./img/多角度拍摄/角度1/a1_17.jpg')
    # train=cv2.imread('./img/多角度拍摄/角度2/IMG_1805.jpg')
    train=cv2.imread('./img/front/pic5.jpg')
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
    cv2.imwrite('./v2/match.jpg', img3)

    im_zeros = np.zeros(query.shape, np.uint8)
    for point in kp1:
        cv2.circle(im_zeros, (int(point.pt[0]), int(point.pt[1])), 10, (255, 255, 255), -1)

    im_zeros = cv2.dilate(im_zeros, kernel2, iterations=5 )
    cv2.imwrite('./v2/points.jpg', im_zeros)

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
    cv2.imwrite('./v2/sobelx.jpg', cutImg)
    threshold, cutImg = cv2.threshold(cutImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cutImg = cv2.adaptiveThreshold(cutImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./v2/cutadapt.jpg', cutImg)
    # cutImg = cv2.morphologyEx(cutImg, cv2.MORPH_OPEN, kernel4)
    # cv2.imwrite('./v2/cutopened.jpg', cutImg)
    # cutImg = cv2.erode(cutImg, kernel3, iterations=1)
    # cv2.imwrite('./v2/cuterode.jpg', cutImg)
    cutImg = cv2.dilate(cutImg, kernel4, iterations=2)
    cv2.imwrite('./v2/cutdilate.jpg', cutImg)

    cutImg = cutImage(cutImg, 'numZone.jpg', instrumentArea)

    # 倾斜矫正
    correctImg = correctAngle(cutImg)
    return separateNum(correctImg)




















