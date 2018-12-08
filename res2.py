'''
本版本使用简单的腐蚀膨胀，直接获取最大区域为数字区域。

'''

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
import sys
import os
import platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(65, 65))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 4))


kernel85 = cv2.getStructuringElement(cv2.MORPH_RECT, (82, 82))
kernel80 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))

kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29))
# kernel35 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
kernel40 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))


def imwrite(_path,_img):
    cv2.imencode('.jpg',_img)[1].tofile(_path)
    print('img',_path)



def cutImage(amask, origin,model=1):
    '''

    :param amask:
    :param origin:
    :param model: model!=1 return all the rectangles' [x1,y1,x2,y2]
    :return:
    '''
    _, contours, hierarchy = cv2.findContours(amask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours)==0:
        print('err', '未找到可数字区域')
        sys.exit(0)

    # 绘制轮廓
    if model==1:
        c = sorted(contours, key=cv2.contourArea, reverse=True)
    else:
        c = contours
    cont_img = origin.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    # 绘制结果
    # imwrite('./pre_model/3_1red_contours.jpg', cont_img)
    block_arr = []

    sets = []
    for cur in c:
        numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(cur)))
        Xs = [i[0] for i in numBox]
        Ys = [i[1] for i in numBox]
        x1 = min(Xs) if min(Xs)>0 else 0
        x2 = max(Xs)
        y1 = min(Ys) if min(Ys)>0 else 0
        y2 = max(Ys)

        hight = y2 - y1
        width = x2 - x1
        if hight*width> cv2.countNonZero(amask)/12:
            cut = origin[y1:y1 + hight, x1:x1 + width]
            sets.append([x1, y1, x2, y2])
            block_arr.append(cut)
    if model==1:
        return block_arr[0]
    elif model==2:
        tmp=[]
        for s in sets:
            if len(tmp)==0:
                tmp.append([s])
            else:
                ind=0
                for t in tmp:
                    ind+=1
                    if (s[3]+s[1])/2 >t[len(t)-1][1] and (s[3]+s[1])/2 <t[len(t)-1][3]:
                        t.append(s)
                        break
                    elif ind==len(tmp):
                        tmp.append([s])
                        break
        # print('sets',sets)
        tmp = sorted(tmp, key=lambda x: (x[0][1]))
        sets=[]
        for ta in tmp:
            # print('ta',ta)
            td = sorted(ta,key=lambda x:(x[2]))
            sets.append(td)

        print('sorted', sets)
        return sets
    else:
        return sorted(sets,key=lambda x:(x[2]))




def get_contours_area(mask, cut):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    cut_arr=[]
    rect_arr=[]

    for c in cnts:
        rect = cv2.boundingRect(c)
        rect_arr.append(rect)

    rect_arr = sorted(rect_arr, key=lambda a: a[0])
    for [x, y, w, h] in rect_arr:
        new_img = cut[y:y + h, x:x + w]
        small_mask = mask[y:y + h, x:x + w]
        if cv2.countNonZero(new_img)/cv2.countNonZero(small_mask)<0.5:
            cut_arr.append(new_img)



    return cut_arr








def bright_avg(src):
    '''
    计算亮度
    :param src: 灰度图
    :return:
    '''
    avg,stdd = cv2.meanStdDev(src)
    print(avg,stdd)
    return avg[0][0],stdd[0][0]


def split_light(src,index=0):
    # src = calc_equalize(src)
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)
    print('亮度均值：',np.mean(V))
    avg_light = np.mean(V)
    low=254
    if avg_light >=100:
        low=250
    elif avg_light >=125:
        low = 180

    if p1 != -1:
        low = p1
    print('当前亮度提取低阈值为' + str(low))
    mask4_l = cv2.inRange(V, low, 255)
    # imwrite('./res1/5_light'+str(index)+'.jpg', mask4_l)
    imwrite(dir_path+'/1_light'+str(index)+'.jpg', mask4_l)
    return mask4_l




def getNumArea(pth):
    # img1=cv2.imread("./img/front/dark/d8.jpg")
    if platform.system() == 'Windows':
        # path = path.encode('utf-8')
        # path = path.decode()
        img1 = cv2.imdecode(np.fromfile(pth, dtype=np.uint8),cv2.IMREAD_COLOR)
        # img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    else:
        img1 = cv2.imread(pth)

    # img1=cv2.imread("./img/front/pic10.jpg")
    # img1=cv2.imread("./img/多角度拍摄/角度2/a2_12.jpg")
    # canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    if img1 is None:
        print('err','图片路径有误'+pth)
    imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # imgGray = cv2.bilateralFilter(imgGray, 9, 70, 70)
    # imgGray = cv2.GaussianBlur(imgGray, (7, 7), 1)
    # bright_avg(imgGray)
    # imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
    # imwrite(dir_path+'/0_adapt.jpg',imgAdapt)
    #
    # opened = cv2.morphologyEx(imgAdapt, cv2.MORPH_OPEN,kernel1)
    # imwrite(dir_path+'/0_opened1.jpg',opened)
    #
    # opened = cv2.erode(opened, kernel3, iterations=4)
    # opened = cv2.dilate(opened, kernel2, iterations=5)
    # imwrite(dir_path+'/0_opened2.jpg', opened)
    threshold, imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imwrite(dir_path+'/1_imgOtsu.jpg', imgOtsu)
    light_mask = split_light(img1)

    img_tmp = cv2.bitwise_and(imgOtsu,light_mask)
    img_tmp = cv2.erode(img_tmp, kernel3, iterations=1)
    # imwrite(dir_path+'/0_img_tmp.jpg', img_tmp)

    img_mask = cv2.dilate(img_tmp, kernel2, iterations=5)
    imwrite(dir_path+'/1_img_tmp.jpg', img_mask)

    cut_img_arr=get_contours_area(img_mask,img_tmp)
    for i in range(len(cut_img_arr)):
        imwrite(dir_path+'/1_cut_img'+str(i)+'.jpg', cut_img_arr[i])
    return cut_img_arr


def load_cnn():
    model_path = './myCNN.h5'
    K.clear_session()  # Clear previous models from memory.
    # cnn_model = load_model(model_path)
    try:
        cnn_model = load_model(model_path)
    except:
        print('err', '程序目录下找不到cnn模型')
        sys.exit(0)
    return cnn_model

def convert2Num(onehot):
    '''

    :param onehot: type nparray only
    :return:
    '''

    p= np.where(onehot==np.max(onehot))
    # print(onehot)
    # print(p)
    return str(int(p[1][0]))

def correctAngle(src,index):
    origin_src = src.copy()
    if src.shape[1]<300:
        default_v = 80
    else:
        default_v =150
    canny = cv2.Canny(cv2.GaussianBlur(src, (5, 5), 0), 0, default_v)
    imwrite(dir_path+'/4_canny'+str(index)+'.jpg', canny)
    canny = cv2.dilate(canny, kernel3, iterations=1)
    # imwrite(dir_path+'/7_cannyDilate'+str(index)+'.jpg', canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 50)
    if lines is None or len(lines)==0:
        print('err', '未检测到直线，倾斜矫正失败！')
        sys.exit(0)
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

    correct = cv2.warpAffine(origin_src, M, (width, height))

    imwrite(dir_path+'/4_correct'+str(index)+'.jpg', correct)
    return correct

def makedilateMask(src,index):
    wid = src.shape[1]
    kel = kernel30
    if wid < 620 and wid > 200:
        kel= kernel30
    elif wid > 620 and wid < 1000:
        kel= kernel40
    elif wid > 1000 and wid<1400:
        kel= kernel2
    elif wid > 1400:
        kel = kernel85
    elif wid < 200:
        kel= kernel5
    if p2!=-1:
        kel = cv2.getStructuringElement(cv2.MORPH_RECT, (p2, p2))

    mask = cv2.dilate(src,kel)
    imwrite(dir_path+'/5_mask'+str(index)+'.jpg',mask)
    return mask




def processMain(pth,outPath = './result.txt'):
    print('1.获取数字区域')
    cut_img_arr = getNumArea(pth)
    if len(cut_img_arr)==0:
        print('err', '未检测到数字区域')
        sys.exit(0)

    b_index = 0
    # num_blocks = []
    dot_img = cv2.imread('./dot1.jpg', 0)
    # dot_img = cv2.GaussianBlur(cv2.resize(dot_img,(10,10)),(3,3),1)
    #
    # imwrite('./dot1.jpg',dot_img)
    if dot_img is None:
        print('err','缺失小数点模板')
        sys.exit(0)
    d_w, d_h = dot_img.shape[::-1]
    dots = []
    for i in range(3):
        dots.append(cv2.resize(dot_img, (int(10 / (1 + 0.1 * i)), int(10 / (1 + 0.1 * i)))))
    for i in range(2):
        dots.append(cv2.resize(dot_img, (int(10 * (1 + 0.1 * i)), int(10 * (1 + 0.1 * i)))))
    rate = 0.6
    print('2.加载cnn模型')
    cnn = load_cnn()

    result_arr=[]
    for cut_img in cut_img_arr:
        b_index += 1
        print('倾斜矫正')
        correct = correctAngle(cut_img,b_index)

        # lines = split_line(correct)
        # for one_line in lines:
        #     num_mask = makedilateMask(one_line)
        #     nums_sets = cutImage(num_mask,one_line,-1)
        #     ind=0
        #     for one_set in nums_sets:
        #         print(one_set)
        #         a_num = one_line[one_set[1]:one_set[3], one_set[0]:one_set[2]]
        #         imwrite('./res1/9_a_num' + str(ind) + '.jpg', a_num)
        #         ind+=1
        print('膨胀数字模板')
        num_mask = makedilateMask(correct,b_index)
        nums_sets = cutImage(num_mask, correct, 2)
        if len(nums_sets)==0:
            print('err', '未找到数字区域')
            sys.exit(0)
        ind = 0

        for line in nums_sets:
            line_dots = []
            res_list = []
            for one_set in line:
                a_num = correct[one_set[1]:one_set[3], one_set[0]:one_set[2]]
                # imwrite('./res1/9_a_num' + str(ind) + '.jpg', a_num)
                a_num = cv2.GaussianBlur(cv2.resize(a_num, (32, 64)), (3, 3), 1)
                tmp_dot_max = 0
                x = a_num.astype(float)
                x *= (1. / 255)
                x = np.array([x])
                x = x.reshape(1, 64, 32, 1)
                result = cnn.predict(x)
                print('数字：', convert2Num(result))
                res_list.append(convert2Num(result))
                for i in range(5):
                    res = cv2.matchTemplate(a_num, dots[i], cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= rate)
                    # print(np.max(res))
                    # print(loc)
                    for pt in zip(*loc[::-1]):
                        if pt[0] >= 16 and pt[1] >= 45:
                            if res[pt[1]][pt[0]] > tmp_dot_max:
                                tmp_dot_max = res[pt[1]][pt[0]]
                            cv2.rectangle(a_num, pt, (pt[0] + d_w, pt[1] + d_h), 255, 1)
                line_dots.append(tmp_dot_max)
                imwrite(dir_path+'/6_a_num' + str(b_index) + '_' + str(ind) + '.jpg', a_num)
                ind += 1
            print('小数点arr:', line_dots)
            dot_ind = np.where(line_dots == np.max(line_dots))[0][0]
            if line_dots[dot_ind] != 0:
                # print('小数点：',dot_ind)
                res_list.insert(dot_ind + 1, '.')
            else:
                print('区域内小数点检测失败!')

            result = ''.join(res_list)
            print('res', result)
            result_arr.append(result)
    f = open(outPath, 'w')
    f.write('\n' + pth)
    for i in range(len(result_arr)):
        f.write(' 结果'+str(i) + '：' + str(result_arr[i]))
    f.close()
    print('结果输出在' + outPath)


if __name__ == '__main__':
    args = sys.argv[1:]
    # print(args)
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--src', help='input image path')
    # parser.add_argument('--out', help='path of result txt ')
    # args =  parser.parse_args(args)
    # print(args)


    try:
        args.index('p1')
    except:
        p1 = -1
    else:
        p1 =int(args[args.index('p1') + 1])

    try:
        args.index('p2')
    except:
        p2 = -1
    else:
        p2 = int(args[args.index('p2') + 1])

    try:
        args.index('p3')
    except:
        p3 = 100
    else:
        p3 = int(args[args.index('p3') + 1])

    try:
        args.index('p4')
    except:
        p4 = 8
    else:
        p4 = int(args[args.index('p4') + 1])

    try:
        args.index('p5')
    except:
        p5 = 4
    else:
        p5 = int(args[args.index('p5') + 1])

    try:
        args.index('p6')
    except:
        p6 = 10
    else:
        p6 = int(args[args.index('p6') + 1])

    kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (p5, p5))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (p6, p6))

    try:
        args.index('out')
    except:
        if len(args) == 0:
            print('请输入图片路径')
            sys.exit(0)
        elif len(args) >= 1:
            print('图片路径：', args[0])
            img_path = args[0]
            dir_path_arr =img_path.split(os.path.sep)[:-1]
            filename = img_path.split(os.path.sep)[-1].split('.')[0]
            dir_path = os.path.sep.join(dir_path_arr) + os.path.sep + filename
            isExists = os.path.exists(dir_path)
            if not isExists:
                os.makedirs(dir_path)
                # os.makedirs(dir_path+'/num1')
                # os.makedirs(dir_path+'/num2')
            processMain(img_path)
    else:
        out_path = args[args.index('out') + 1]
        print('图片路径：', args[0])
        print('结果保存路径：', out_path)
        img_path = args[0]
        dir_path_arr = img_path.split(os.path.sep)[:-1]
        filename = img_path.split(os.path.sep)[-1].split('.')[0]
        dir_path = os.path.sep.join(dir_path_arr)+os.path.sep+filename
        isExists = os.path.exists(dir_path)
        if not isExists:
            os.makedirs(dir_path)
            # os.makedirs(dir_path + '/num1')
            # os.makedirs(dir_path + '/num2')
        processMain(img_path, out_path)
