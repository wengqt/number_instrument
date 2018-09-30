import cv2
import numpy as np
import softmax

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

##
# @param findContoursImg 膨胀之后用于找矩形轮廓的图
#        dst 目标生成的文件名
#        origin 用于切割的原图
def cutImage(mask, origin):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    cont_img = origin.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    # 绘制结果
    cv2.imwrite('./v6/3_1red_contours.jpg', cont_img)
    block_arr = []
    for cur in c:
        numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(cur)))
        Xs = [i[0] for i in numBox]
        Ys = [i[1] for i in numBox]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cut = origin[y1:y1 + hight, x1:x1 + width]
        block_arr.append(cut)
    return block_arr



def correctAngle(src):
    canny = cv2.Canny(cv2.GaussianBlur(src, (5, 5), 0), 50, 100)
    cv2.imwrite('./v6/numZonecanny.jpg', canny)
    canny = cv2.dilate(canny, kernel3, iterations=1)
    cv2.imwrite('./v6/numZonecannyDilate.jpg', canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 40)
    numLine = 0
    angle = 0
    (height,width) = src.shape

    for ls in lines:
        for line in ls:
            rho = line[0]
            theta = line[1]
            if abs(np.pi / 2 - theta) < np.pi / 6:
                print('this angle : ', theta * 180 / np.pi)
                numLine = numLine + 1
                angle = angle + theta

            # if angle > (np.pi / 2):
            #     angle = angle - np.pi

        # print('angle : %f, split : %f' % (angle, numLine))

    averageAngle = (angle / float(numLine)) * 180 / np.pi
    print('averageAngle : %f' % averageAngle)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), averageAngle - 90, 1.0)

    correct = cv2.warpAffine(src, M, (width, height))

    cv2.imwrite('./v6/numZonecorrect.jpg', correct)
    return correct



def splitNum(num_Img, index,min_range=0):

    cv2.imwrite('./v6/split/split' + str(int(index)) + 'Img.jpg', num_Img)
    num1closed = cv2.morphologyEx(num_Img, cv2.MORPH_CLOSE, kernel3)
    cv2.imwrite('./v6/split/split' + str(int(index)) + 'closed.jpg', num1closed)

    (hI1, wI1) = num1closed.shape
    vertical1 = [0 for z in range(0, wI1)]

    for i in range(0, wI1):  # 遍历一lie
        for j in range(0, hI1):  # 遍历一hang
            if num1closed[j, i] != 0:
                vertical1[i] += 1

    newHorizon = np.zeros([hI1, wI1], np.uint8)
    #
    for i in range(0, wI1):
        for j in range(0, vertical1[i]):
            newHorizon[j, i] = 255

    cv2.imwrite('./v6/newHorizon.jpg', newHorizon)
    num1points = []

    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i if i > 0 else 0
                num1points.append(t)
                getLineZone2(src, t, n2)
                break

    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i  if i <= n2 else n2
                num1points.append(t)
                getLineZone1(src, t, n2)
                break
            elif i >= n2 - 1:
                num1points.append(n2 - 1)

    getLineZone1(vertical1, 0, wI1)

    tmp = []
    for i in range(0, len(num1points), 2):
        if num1points[i + 1] - num1points[i] > min_range:
            tmp.append(num1points[i])
            tmp.append(num1points[i + 1])

    num1points = tmp
    print('num_border', num1points)
    maxW = 0
    for i in range(0, len(num1points), 2):
        maxW = num1points[i + 1] - num1points[i] if maxW < num1points[i + 1] - num1points[i] else maxW

    allImgs = []
    for i in range(0, len(num1points), 2):
        # num1points[i] = num1points[i + 1] - maxW
        cv2.imwrite('./v6/split/split' + str(index) + '_' + str(int(i / 2)) + '.jpg',
                    num_Img[:, num1points[i]:num1points[i + 1]])
        allImgs.append(num_Img[:, num1points[i]:num1points[i + 1]])
    return allImgs

def separateNum(correct_src,min_range=0):
    ##分行
    (h1, w1) = correct_src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if correct_src[i, j] != 0:
                horizon[i] += 1

    line_border = []

    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i if i > 0 else 0
                line_border.append(t)
                getLineZone2(src, t, n2)
                break

    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i if i <= n2 else n2
                line_border.append(t)
                getLineZone1(src, t, n2)
                break
            elif i >= n2 - 1:
                line_border.append(n2 - 1)

    getLineZone1(horizon, 0, h1)
    tmp = []
    for i in range(0, len(line_border), 2):
        if line_border[i + 1] - line_border[i] > min_range:
            tmp.append(line_border[i])
            tmp.append(line_border[i + 1])

    line_border=tmp
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

            dot_line = cv2.HoughLines(oneimg, 1, np.pi / 180, 10)
            l_num = 0
            ang = 0
            if dot_line is not None:
                for ls in dot_line:
                    for line in ls:
                        rho = line[0]
                        theta = line[1]
                        if abs(theta) < np.pi / 6:
                            print('this angle : ', theta * 180 / np.pi)
                            l_num = l_num + 1
                            ang = ang + theta
                avg_angle = ang / float(l_num) * 180 / np.pi
            else:
                avg_angle =0
            print('竖直夹角： ', avg_angle)
            oneimg_cp = oneimg.copy()
            # oneimg_cp = cv2.dilate(oneimg_cp, kernel3, iterations=2)
            one_h, one_w = oneimg_cp.shape[:2]
            M_1 = cv2.getRotationMatrix2D((one_w / 2, one_h / 2), avg_angle, 1.0)

            oneimg_cp = cv2.warpAffine(oneimg_cp, M_1, (one_w, one_h))

            cv2.imwrite('./v6/split/correct' + str(j) + '_' + str(one_index) + '.jpg', oneimg_cp)
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
            cv2.imwrite('./v6/res/' + str(11) + '_' + str(j) + '_' + str(one_index) + '.jpg', oneimg)
            arr.append('./v6/res/' + str(11) + '_' + str(j) + '_' + str(one_index) + '.jpg')
        imgList.append(arr)
    print('dots:',dots)
    return imgList, dots


def getNumZone(src,dilateORnot=1):
    img_o = cv2.bilateralFilter(src, 9, 50,50)


    img_hsv = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    # Lowerred0 = np.array([155, 43, 46])
    # Upperred0 = np.array([180, 255, 255])
    # mask1 = cv2.inRange(img_hsv, Lowerred0, Upperred0)
    # Lowerred1 = np.array([0, 43, 46])
    # Upperred1 = np.array([11, 255, 255])
    # mask2 = cv2.inRange(img_hsv, Lowerred1, Upperred1)  # 将红色区域部分归为全白，其他区域归为全黑
    # Lowerred2 = np.array([35, 43, 46])
    # Upperred2 = np.array([77, 255, 255])
    # mask3 = cv2.inRange(img_hsv, Lowerred2, Upperred2)
    # mask4_l = mask1 + mask2 + mask3
    # cv2.imwrite('./v6/masks.jpg',mask4_l)


    mask4_l = cv2.inRange(V, 250, 255)
    cv2.imwrite('./v6/masks.jpg', mask4_l)
    if dilateORnot == 1:
        mask4 = cv2.dilate(mask4_l, kernel5,iterations=3)
        cv2.imwrite('./v6/masks_1.jpg', mask4)
        return mask4_l,mask4
    else:
        return mask4_l






def fillAnd(src1,src2,mask=None):
    and_img =  cv2.bitwise_and(src1, src2,mask =mask)
    return and_img





def matchArea(src1,src2):
    # 暴力匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(src1, None)
    kp2, des2 = orb.detectAndCompute(src2, None)
    # 针对ORB算法 NORM_HAMMING 计算特征距离 True判断交叉验证
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 特征描述子匹配
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    img3 = cv2.drawMatches(src1, kp1, src2, kp2, matches[:200], None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite('./v6/1_match.jpg', img3)

    img_points = np.zeros(src1.shape, np.uint8)
    for point in kp1:
        cv2.circle(img_points, (int(point.pt[0]), int(point.pt[1])), 30, (255, 255, 255), -1)

    img_points = cv2.morphologyEx(img_points, cv2.MORPH_OPEN, kernel5, iterations=5)
    img_points = cv2.dilate(img_points, kernel1)
    cv2.imwrite('./v6/2_points.jpg', img_points)

    return img_points

def learnNums():
    softmax_learn = softmax.Softmax()
    trainDigits, trainLabels = softmax_learn.loadData('./train')
    softmax_learn.train(trainDigits, trainLabels, maxIter=100)  # 训练
    return softmax_learn


if __name__ == '__main__':
    query = cv2.imread('./img/query.jpg')

    query = cv2.cvtColor(cv2.GaussianBlur(query, (7, 7), 1), cv2.COLOR_RGB2GRAY)
    # query = cv2.equalizeHist(query)
    # train = cv2.imread(dir)
    train_o = cv2.imread('./img/im2.jpg') #待检测图
    useImg = cv2.imread('./img/im1.jpg')  #效果良好的图进行与操作

    train = cv2.cvtColor(cv2.GaussianBlur(train_o, (7, 7), 1), cv2.COLOR_RGB2GRAY)

    # train = cv2.equalizeHist(train)
    # train=cv2.imread('./img/多角度拍摄/角度4/a11.jpg')
    # train = cv2.imread('./img/cam1.png')
    # train=cv2.imread('./img/多角度拍摄/角度2/IMG_1807.jpg')
    # train=cv2.imread('./img/front/pic5.jpg')
    # train = cv2.equalizeHist(train)
    # train=cv2.imread('./img/front/pic19.jpg')
    # train=cv2.imread("./img/front/dark/d7.jpg")
    # train1 = cv2.cvtColor(train, cv2.COLOR_RGB2GRAY)
    # train1 = cv2.equalizeHist(train1)

    im_zeros = matchArea(train,query)
    instrumentArea = fillAnd(train_o,train_o,im_zeros)
    cv2.imwrite('./v6/3_cut1.jpg', instrumentArea)
    useArea = fillAnd(useImg,useImg,im_zeros)
    cv2.imwrite('./v6/3_cut2.jpg', useArea)


    #亮度提取数字区域：
    instru_light,instru_dilate = getNumZone(instrumentArea)
    use_light,use_dilate = getNumZone(useArea)

    num_area = fillAnd(instrumentArea,instrumentArea,use_dilate)
    cv2.imwrite('./v6/4_num.jpg', num_area)
    #普通分割
    num_area_gray = cv2.cvtColor(cv2.GaussianBlur(num_area,(7,7),1), cv2.COLOR_RGB2GRAY)
    # threshold, img_adp = cv2.threshold(num_area_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold,img_adp = cv2.threshold(num_area_gray, 140, 255, cv2.THRESH_BINARY)
    # img_adp = cv2.adaptiveThreshold(num_area_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)

    cv2.imwrite('./v6/4_num_adp.jpg', img_adp)

    blocks = cutImage(use_dilate,img_adp)

    for i in range(len(blocks)):
        one_area = blocks[i]
        cv2.imwrite('./v6/5_area'+str(i)+'.jpg', one_area)
    corr_img =  correctAngle(blocks[2])
    nums, dot_p =  separateNum(corr_img,10)

    slearn = learnNums()
    




