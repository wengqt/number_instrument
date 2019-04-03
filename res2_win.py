
# coding=utf-8

from keras import backend as K
import numpy as np
from keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import cv2
import os
import platform
import sys


'''
这个版本为不使用其他图片，直接识别的版本。
使用ssd模型版本
'''





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (66, 66))
kernel85 = cv2.getStructuringElement(cv2.MORPH_RECT, (82, 82))
kernel80 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29))
# kernel35 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
kernel40 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))



def imwrite(_path,_img):
    cv2.imencode('.jpg',_img)[1].tofile(_path)
    print('img',_path)



def check_arr(arr):
    line_id = 0
    LINE_DOT = [None for x in range(len(arr))]
    for line in arr:
        # line = sorted(line, key=lambda x: (x[2]))
        h_arr=[x[3]-x[1] for x in line]#高度数组
        h_avg = sum(h_arr)/len(h_arr)
        h_avg = h_avg-h_avg*0.2
        w_arr = [x[2]-x[0] for x in line]#宽度数组
        base_w = max(w_arr)/2
        center_line = [(x[3]+x[1])/2 for x in line]
        base_center_min = sum(center_line)/len(center_line) - h_avg*0.2
        base_center_max = sum(center_line)/len(center_line) + h_avg*0.2
        for i in range(len(w_arr)):
            # print(base_center_min<center_line[i]<base_center_max,w_arr[i] <base_w,w_arr[i],base_w)
            # print(h_arr[i] , h_avg-h_avg*0.15)
            # print(h_arr[i]>w_arr[i],base_center_min<center_line[i]<base_center_max,h_arr[i]<h_avg-h_avg*0.15)
            if h_arr[i]>w_arr[i] and base_center_min<center_line[i]<base_center_max and h_arr[i]<h_avg-h_avg*0.15:
                line[i]=None

        for i in range(len(h_arr)):
            # print(h_arr[i]<h_avg-h_avg*0.2)
            if center_line[i]>base_center_max and h_arr[i]<h_avg-h_avg*0.15:
                # print(line[i])
                if i ==0:
                    LINE_DOT[line_id]=0
                    line[0]=None
                elif len(h_arr)>0 and i<=len(h_arr)-1:
                    j_ =i-1
                    while j_>=0:
                        if line[j_] is not None :
                            # print(line[i][0]-line[j_][2])
                            if line[i][0]-line[j_][2]<12:
                                LINE_DOT[line_id] = i

                                break
                        j_ -= 1
                    line[i] = None
        ind=0
        while ind<len(line) :
            if line[ind] is None:
                del line[ind]
            else:
                ind += 1

        line_id += 1

    return arr, LINE_DOT

def cutImage(amask):
    '''

    :param amask:
    :param origin:
    :param model: model!=1 return all the rectangles' [x1,y1,x2,y2]
    :return:
    '''
    _, c, hierarchy = cv2.findContours(amask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓

    # 绘制结果
    # imwrite('./pre_model/3_1red_contours.jpg', cont_img)

    sets = []
    for cur in c:
        x, y, w, h = cv2.boundingRect(cur)
        # print(w*h,cv2.countNonZero(amask)/50)
        if w*h> cv2.countNonZero(amask)/50:
            sets.append([x, y, x + w, y + h])

    sets = sorted(sets, key=lambda x: (x[2]))
    sets = sorted(sets, key=lambda x: (x[3]))
    tmp=[]
    for s in sets:
        if len(tmp)==0:
            tmp.append([s])
        else:
            ind=0
            for t in tmp:
                ind+=1
                # print(s)
                # print((s[3]+s[1])/2 >t[len(t)-1][1],(s[3]+s[1])/2 <t[len(t)-1][3])
                if ((s[3]+s[1])/2 >t[len(t)-1][1]-15 and (s[3]+s[1])/2 <t[len(t)-1][3]+15) or(s[3]<t[len(t)-1][3]+20 and s[1]<t[len(t)-1][3]):
                    t.append(s)
                    break
                elif ind==len(tmp):
                    tmp.append([s])
                    break
    # print('sets',tmp)



    tmp = sorted(tmp, key=lambda x: (x[0][3]))
    sets=[]
    for ta in tmp:
        # print('ta',ta)
        if len(ta)>1:
            td = sorted(ta,key=lambda x:(x[2]))
            sets.append(td)

    sets, Line_Dot = check_arr(sets)
    # print('sorted', sets)
    return sets,Line_Dot



def cutBlocks(mask,origin):
    _, c, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = origin.copy()
    # for i in range(len(c)):
    #     # rect = cv2.minAreaRect(c[i])
    #     x, y, w, h = cv2.boundingRect(c[i])
    #     # box = np.int0(cv2.boxPoints(rect))
    #     # cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    #     cv2.rectangle(cont_img,(x,y),(x+w,y+h),(0,0,255),2)
    # block_arr = []
    sets = []
    for cur in c:
        x, y, w, h = cv2.boundingRect(cur)
        # if h*w> cv2.countNonZero(mask)/12:
            # cut = origin[y:y + h, x:x + w]
        sets.append([x, y, x+w, y+h])
            # block_arr.append(cut)

    sets = sorted(sets, key=lambda x: (x[3]))
    return sorted(sets, key=lambda x: (x[2]))

# def matchArea(src1,src2):
#     src1 = cv2.cvtColor(cv2.GaussianBlur(src1, (7, 7), 1), cv2.COLOR_RGB2GRAY)
#     src2 = cv2.cvtColor(cv2.GaussianBlur(src2, (7, 7), 1), cv2.COLOR_RGB2GRAY)
#     # 暴力匹配
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(src1, None)
#     kp2, des2 = orb.detectAndCompute(src2, None)
#     # 针对ORB算法 NORM_HAMMING 计算特征距离 True判断交叉验证
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # 特征描述子匹配
#     matches = bf.match(des1, des2)
#
#     matches = sorted(matches, key=lambda x: x.distance)
#     # print(len(matches))
#     img3 = cv2.drawMatches(src1, kp1, src2, kp2, matches[:200], None, matchColor=(0, 255, 0), flags=2)
#     imwrite(dir_path+'/1_match.jpg', img3)
#
#     img_points = np.zeros(src1.shape[:2], np.uint8)
#     for point in kp1:
#         cv2.circle(img_points, (int(point.pt[0]), int(point.pt[1])), 30, 255, -1)
#
#     # img_points = cv2.morphologyEx(img_points, cv2.MORPH_OPEN, kernel5, iterations=5)
#     img_points = cv2.dilate(img_points, getKernel_2(src1.shape[1]), iterations=4)
#     imwrite(dir_path+'/2_points.jpg', img_points)
#
#     return img_points

def predict_num_area(src):
    '''
    :param src:opencv读取的图
    :return: scale:缩放倍率，mask:数字区域的全白图,all_blocks:矩形坐标
    '''

    # model_path = 'ssd7_pascal_07_epoch-17_loss-0.8387_val_loss-0.8608.h5'
    model_path = 'ssd7_4.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session()  # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'compute_loss': ssd_loss.compute_loss})
    # img_height = 341
    # img_width = 256
    img_height = 256
    img_width = 456
    scale_y = src.shape[0]/img_height
    scale_x = src.shape[1]/img_width





    normalize_coords = True
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.

    # We'll only load one image in this example.
    # filename = '../NumInstrument/img/im3.JPG'
    # filename='../ssd_trains/JPEGImages/image1024.JPG'

    # img = cv2.imread(filename)



    if scale_x>scale_y:
        scale = scale_x
        real_w = img_width
        real_h = int(src.shape[0] / scale)
        black_img = np.zeros((img_height,img_width,3),np.uint8)
        img = cv2.resize(src, (real_w, real_h))

        t_ = int((img_height-real_h)/2)
        black_img[t_:t_+real_h,:]=img
    else:
        scale = scale_y
        real_w = int(src.shape[1] / scale)
        real_h = img_height
        black_img = np.zeros((img_height, img_width, 3), np.uint8)
        img = cv2.resize(src, (real_w, real_h))
        t_ = int((img_width - real_w) / 2)
        black_img[:,t_:t_ + real_w] = img

    orig_images.append(black_img)
    img = cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    # 4: Decode the raw predictions in `y_pred`.
    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=True,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_decoded[0])
    mask = np.zeros((img_height,img_width),np.uint8)

    mask_sets = []
    for box in y_pred_decoded[0]:
        xmin = int(box[2]-5) if int(box[2]-5)>=0 else 0
        ymin = int(box[3])
        xmax = int(box[4] + 7)
        ymax = int(box[5] + 5 )
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
        mask_sets.append([xmin,ymin,xmax,ymax])
        cv2.rectangle(input_images[0], (xmin, ymin), (xmax, ymax), 255, 2)

    input_images[0] = cv2.cvtColor(input_images[0], cv2.COLOR_RGB2BGR)

    # imwrite(dir_path + '/0_black_img.jpg', black_img)
    # mask = cv2.dilate(mask, kernel4)

    imwrite(dir_path+'/0_pre_mask.jpg', input_images[0])
    # imwrite(dir_path+'/4_pre_res' + str(index) + '.jpg', input_images[0])

    all_blocks = cutBlocks(mask,input_images[0])
    # print(all_blocks,scale)
    if len(all_blocks)==0:
        print('err', '未找到数字区域')
        sys.exit(0)

    tmp=[]

    for abox in all_blocks:
        if scale_x>scale_y:
            aa=[int((abox[0])*scale),int((abox[1]-t_)*scale),int(abox[2]*scale),int((abox[3]-t_)*scale)]
        else:
            aa=[int((abox[0]-t_)*scale),int(abox[1]*scale),int((abox[2]-t_)*scale),int(abox[3]*scale)]
        tmp.append(aa)
    all_blocks = tmp

    tmp = []
    for abox in mask_sets:
        if scale_x>scale_y:
            aa=[int((abox[0])*scale),int((abox[1]-t_)*scale),int(abox[2]*scale),int((abox[3]-t_)*scale)]
        else:
            aa=[int((abox[0]-t_)*scale),int(abox[1]*scale),int((abox[2]-t_)*scale),int(abox[3]*scale)]
        tmp.append(aa)
    mask_sets = tmp

    mask_img = np.zeros((src.shape[0],src.shape[1]),np.uint8)
    for rect in mask_sets:
        [x1,y1,x2,y2] = rect
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), 255, -1)

    K.clear_session()
    return mask,all_blocks,mask_img


def calc_gamma(src,rate):
    img0 = src


    def gamma_trans(img, gamma):
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 实现映射用的是Opencv的查表函数
        return cv2.LUT(img0, gamma_table)

    img0_corrted = gamma_trans(img0, rate)


    return img0_corrted




def get_light_mask(src,index):
    # src = calc_equalize(src)
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)
    l_d = np.mean(V)
    print('亮度均值：',l_d)

    low=200
    if l_d >=100:
        low=240
    elif l_d>=130:
        low = 250
    elif l_d <50 and l_d>30:
        low = 240
    elif l_d>50 and l_d<100:
        low = 240
    elif l_d<30:
        low = 150

    if p1!=-1:
        low = p1
    print('当前亮度提取低阈值为' + str(low))
    mask4_l = cv2.inRange(V, low, 255)
    imwrite(dir_path+'/3_light'+str(index)+'.jpg', mask4_l)

    return mask4_l,l_d

def calc_equalize(src):
    new_img = []
    sp = cv2.split(src)
    for i in range(3):
        new_img.append(cv2.equalizeHist(sp[i]))
    new_img = np.array(new_img)
    new_img = cv2.merge(new_img)
    imwrite(dir_path+'/0_make_equalize.jpg', new_img)
    return new_img




def adapt_otsu(src,index):
    train_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    threshold, bina2 = cv2.threshold(train_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imwrite(dir_path+'/3_otsu' + str(index) + '.jpg', bina2)
    return bina2


def fillAnd(src1,src2,index,mask=None):
    and_img =  cv2.bitwise_and(src1, src2,mask =mask)
    imwrite(dir_path+'/3_and' + str(index) + '.jpg', and_img)
    return and_img

def getKernel(width):
    if width<500 and width>200:
        return kernel3
    elif width>500 and width<1000:
        return kernel5
    elif width>1000:
        return kernel4
    elif width<200:
        return kernel0

def getKernel_2(wid):
    kel = kernel30
    if wid < 500 and wid > 200:
        kel = kernel30
    elif wid > 500 and wid < 1200:
        kel = kernel40
    elif wid > 1200 and wid < 2000:
        kel = kernel2
    elif wid >2000:
        kel = kernel80
    elif wid < 200:
        kel = kernel5

    return kel

def correctAngle(src,img2=None):
    origin_src = src.copy()
    if src.shape[1]<300:
        default_v = 80
    else:
        default_v =150
    canny = cv2.Canny(cv2.GaussianBlur(src, (5, 5), 0), 0, default_v)

    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    # imwrite(dir_path+'/4_cannyDilate.jpg', canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 100)
    if lines is None:
        return src,canny

    numLine = 0
    angle = 0

    (height,width) = src.shape[:2]
    if lines is None or len(lines)==0:
        print('err', '未检测到直线，倾斜矫正失败！')
        sys.exit(0)

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
    try:
        averageAngle = (angle / float(numLine)) * 180 / np.pi
        print('averageAngle : %f' % averageAngle)
    except:
        return src, canny

    M = cv2.getRotationMatrix2D((width / 2, height / 2), averageAngle - 90, 1.0)

    correct = cv2.warpAffine(origin_src, M, (width, height))
    if img2 is not None:
        img2 = cv2.warpAffine(img2, M, (width, height))
        return correct, canny,img2

    return correct,canny








def get_border(arr,delta):
    '''

    :param arr: input array
    :param delta: the min range between arr item
    :return: border
    '''
    getTwo=False
    border=[]
    for i in range(len(arr)):
        if getTwo==False:
            if arr[i] !=0:
                border.append(i)
                getTwo = True
        else:
            if arr[i] ==0:
                if i>border[len(border)-1]+delta:
                    border.append(i)
                    getTwo = False
                else:
                    del border[len(border)-1]
                    getTwo = False
            if i>=len(arr)-1:
                border.append(i)

    return border



def split_line(src):
    (h1, w1) = src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if src[i, j] == 255:
                horizon[i] += 1

    newHorizon = np.zeros([h1, w1], np.uint8)

    for i in range(0, h1):
        for j in range(0, horizon[i]):
            newHorizon[i, j] = 255

    imwrite(dir_path+'/8_newHorizon.jpg', newHorizon)

    from matplotlib.pyplot import plot, scatter, show
    border_arr = get_border(horizon,10)

    plot(horizon)
    scatter(np.array(border_arr), np.zeros(len(border_arr)), color='blue')
    show()
    # for (pos,val) in mintab:
    aline=[]
    for i in range(0,len(border_arr)-1,2):
        y1 = border_arr[i]
        y2 = border_arr[i+1]
        aline.append(src[y1:y2,:])

    return aline


# def makedilateMask(src,l_d):
#     wid = src.shape[1]
#     kel = kernel30
#     if p2==-1:
#         if wid < 620 and wid > 200 and l_d<100:
#             kel= kernel30
#             print('当前第二个参数值为29')
#         elif wid > 620 and wid < 1000 and l_d<100:
#             kel= kernel40
#             print('当前第二个参数值为40')
#         elif wid > 1000 and wid<1400 and l_d<100:
#             kel= kernel2
#             print('当前第二个参数值为66')
#         elif wid > 1400 and l_d<100:
#             kel = kernel85
#             print('当前第二个参数值为82')
#         elif wid < 200:
#             kel= kernel5
#             print('当前第二个参数值为5')
# 
#         if l_d>100:
    #         if wid > 1400:
    #             kel = cv2.getStructuringElement(cv2.MORPH_RECT, (48, 48))
    #             print('当前第二个参数值为48')
    #         elif wid > 1000 and wid<1400:
    #             kel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    #             print('当前第二个参数值为40')
    #         elif  wid > 620 and wid < 1000:
    #             kel = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33))
    #             print('当前第二个参数值为33')
    #         elif wid < 620 and wid > 200:
    #             kel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    #             print('当前第二个参数值为23')
    # elif p2!=-1:
    #     kel = cv2.getStructuringElement(cv2.MORPH_RECT, (p2, p2))
    #     print('当前第二个参数值为'+str(p2))

    # mask = cv2.dilate(src, kel)


    # return mask



def makedilateMask(src,l_d):
    wid = src.shape[1]
    kel = kernel30
    if p2==-1:
        if wid < 500 and wid > 200 and l_d<100:
            kel=  cv2.getStructuringElement(cv2.MORPH_RECT, (4, 15))
            print('当前第二个参数值为15')
        elif wid > 500 and wid < 1000 and l_d<100:
            kel=  cv2.getStructuringElement(cv2.MORPH_RECT, (4, 24))
            print('当前第二个参数值为20')
        elif wid > 1000 and wid<1400 and l_d<100:
            kel=  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
            print('当前第二个参数值为20')
        elif wid > 1400 and l_d<100:
            kel =  cv2.getStructuringElement(cv2.MORPH_RECT, (8, 26))
            print('当前第二个参数值为20')
        elif wid < 200:
            kel= kernel5
            print('当前第二个参数值为5')

        if l_d>100:
            if wid > 1400:
                kel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 24))
                print('当前第二个参数值为22')
            elif wid > 1000 and wid<1400:
                kel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 22))
                print('当前第二个参数值为16')
            elif  wid > 620 and wid < 1000:
                kel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
                print('当前第二个参数值为14')
            elif wid < 620 and wid > 200:
                kel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 20))
                print('当前第二个参数值为12')
    elif p2!=-1:
        kel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, p2))
        print('当前第二个参数值为'+str(p2))

    mask = cv2.dilate(src, kel)

    ret, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    return mask



def load_cnn():
    # model_path = './myCNN_num_2.h5'
    model_path = './myCNN_new_new.h5'
    K.clear_session()  # Clear previous models from memory.
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
    if int(p[1][0])==10:
        return '-'

    return str(int(p[1][0]))


# def main_process(img_path):
#     query = cv2.imread('./img/query.jpg')
#
#     # query = cv2.cvtColor(cv2.GaussianBlur(query, (7, 7), 1), cv2.COLOR_RGB2GRAY)
#
#     # train_o = cv2.imread('./img/front/pic2.jpg') #待检测图
#     # train_o = cv2.imread('./img/多角度拍摄/角度2/a2_8.jpg')
#     # train_o = cv2.imread("./img/front/dark/d5.jpg")
#     # train_o = cv2.imread('./img/多角度拍摄/角度4/aa11.jpg')
#     # train_o = cv2.imread('./img/im2.jpg')
#     train_o = cv2.imread('./img/多角度拍摄/角度1/a1_19.jpg')
#     # train_o=cv2.imread('./img/front/pic5.jpg')
#     # train_o=cv2.imread(img_path)
#
#     train = train_o
#
#     im_zeros = matchArea(train,query)
#     instru_area = cutImage(im_zeros,train)
#
#     area_mask,blocks_sets = predict_num_area(instru_area)
#
#     b_index=0
#     num_blocks=[]
#     dot_img = cv2.imread('./dot1.jpg',0)
#     # dot_img = cv2.GaussianBlur(cv2.resize(dot_img,(10,10)),(3,3),1)
#     #
#     # imwrite('./dot1.jpg',dot_img)
#     dot_img = cv2.imread('./dot1.jpg',0)
#     d_w, d_h = dot_img.shape[::-1]
#     dots = []
#     for i in range(3):
#         dots.append(cv2.resize(dot_img,(int(10/(1+0.1*i)),int(10/(1+0.1*i)))))
#     for i in range(2):
#         dots.append(cv2.resize(dot_img,(int(10*(1+0.1*i)),int(10*(1+0.1*i)))))
#     rate = 0.6
#
#     cnn = load_cnn()
#     for b in blocks_sets:
#         b_index+=1
#         one_=instru_area[b[1]:b[3],b[0]:b[2]]
#         imwrite(dir_path+'/5_numblock' + str(b_index) + '.jpg', one_)
#         lightmask = get_light_mask(one_,b_index)
#         adapt = adapt_otsu(one_,b_index)
#         res_block = fillAnd(lightmask,adapt,b_index)
#         kernel = getKernel(res_block.shape[1])
#         res_block= cv2.erode(res_block,kernel)
#         imwrite(dir_path+'/6_erode_and' + str(b_index) + '.jpg', res_block)
#         num_blocks.append(res_block)
#         correct =correctAngle(res_block)
#
#         # lines = split_line(correct)
#         # for one_line in lines:
#         #     num_mask = makedilateMask(one_line)
#         #     nums_sets = cutImage(num_mask,one_line,-1)
#         #     ind=0
#         #     for one_set in nums_sets:
#         #         print(one_set)
#         #         a_num = one_line[one_set[1]:one_set[3], one_set[0]:one_set[2]]
#         #         imwrite(dir_path+'/9_a_num' + str(ind) + '.jpg', a_num)
#         #         ind+=1
#
#         num_mask = makedilateMask(correct)
#         nums_sets = cutImage(num_mask, correct, 2)
#         ind = 0
#         for line in nums_sets:
#             line_dots =[]
#             res_list =[]
#             for one_set in line:
#                 a_num = correct[one_set[1]:one_set[3], one_set[0]:one_set[2]]
#                 # imwrite(dir_path+'/9_a_num' + str(ind) + '.jpg', a_num)
#                 a_num =cv2.GaussianBlur(cv2.resize(a_num,(32,64)),(3,3),1)
#                 tmp_dot_max = 0
#                 x = a_num.astype(float)
#                 x *= (1. / 255)
#                 x = np.array([x])
#                 x = x.reshape(1,64,32,1)
#                 result = cnn.predict(x)
#                 print('数字：',convert2Num(result))
#                 res_list.append(convert2Num(result))
#                 for i in range(5):
#                     res = cv2.matchTemplate(a_num, dots[i], cv2.TM_CCOEFF_NORMED)
#                     loc = np.where(res >= rate)
#                     # print(np.max(res))
#                     # print(loc)
#                     for pt in zip(*loc[::-1]):
#                         if pt[0]>=16 and pt[1]>=45:
#                             if res[pt[1]][pt[0]] >tmp_dot_max:
#                                 tmp_dot_max =res[pt[1]][pt[0]]
#                             cv2.rectangle(a_num, pt, (pt[0] + d_w, pt[1] + d_h), 255, 1)
#                 line_dots.append(tmp_dot_max)
#                 imwrite(dir_path+'/9_a_num' + str(ind) + '.jpg', a_num)
#                 ind += 1
#             print('小数点arr:',line_dots)
#             dot_ind = np.where(line_dots == np.max(line_dots))[0][0]
#             if line_dots[dot_ind]!=0:
#                 # print('小数点：',dot_ind)
#                 res_list.insert(dot_ind+1,'.')
#             else:
#                 print('小数点检测失败')
#
#             result = ''.join(res_list)
#             print('最后结果：',result)



def split_light(src,index=0):
    # src = calc_equalize(src)
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)
    print('亮度均值：',np.mean(V))
    avg_light = np.mean(V)
    low=254
    if avg_light >=100:
        low=240
    elif avg_light >=125:
        low = 180
    mask4_l = cv2.inRange(V, low, 255)
    # imwrite(dir_path+'/5_light'+str(index)+'.jpg', mask4_l)
    imwrite(dir_path+'/0_light'+str(index)+'.jpg', mask4_l)
    return mask4_l





def match_dot(dot_cnt,src_cnt):
    out = cv2.matchShapes(dot_cnt, src_cnt, 1 ,0.0)
    return out


def process1(path,outPath = './result.txt'):
    # train_o = cv2.imread('./img/多角度拍摄/角度1/a1_15.jpg')
    # train_o = cv2.imread('./img/多角度拍摄/角度4/aa11.jpg')
    # train_o = cv2.imread('./img/多角度拍摄/角度3/a_8.jpg')
    # train_o = cv2.imread('./img/im3.jpg')
    if platform.system() == 'Windows':
        # path = path.encode('utf-8') 
        # path = path.decode()
        train_o = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
        # img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    else:
        train_o = cv2.imread(path)
    # train_o = cv2.imread(path)
    if train_o is None:
        print('err', '找不到图片'+path)
        sys.exit(0)
    # train_o = cv2.imread("./img/front/dark/d9.jpg")
    # train_o = cv2.imread("./img/front/pic20.jpg")

    # train_gray =cv2.cvtColor(train_o,cv2.COLOR_BGR2GRAY)
    train_1=train_o.copy()
    all_light, stdd = bright_avg(train_o)
    print('gamma校正前的平均亮度', all_light)
    if all_light < 70:
        rate = 0.8 - (60 - all_light) * 0.01
    elif all_light >= 90:
        rate = 2.2 + (all_light - 90) * 0.01
    else:
        rate = 1.5
    print('gamma数值', rate)
    train_o = calc_gamma(train_o, rate)
    imwrite(dir_path + '/0_gamma.jpg', train_o)
    all_light, stdd = bright_avg(train_o)
    print('gamma校正后的平均亮度', all_light)

    #
    # if light>80:
    # light_mask = split_light(train_o,index)
    print('1.获得数字区域')
    area_mask, blocks_sets, mask_img = predict_num_area(train_o)

    train_o = cv2.bitwise_and(train_1,train_1,mask=mask_img)
    # train_o = calc_equalize(train_o)
    # train_o = calc_gamma(train_o, 4.0)
    # adapt = adapt_otsu(train_o, index)
    print('2.加载小数点模板')
    # dot_src = cv2.imread('./dot.jpg', 0)
    #
    # if dot_src is None:
    #     print('err','缺失小数点模板')
    #     sys.exit(0)
    #
    # _, dot_src = cv2.threshold(dot_src, 127, 255,cv2.THRESH_BINARY)
    # _, dot_c, hierarchy = cv2.findContours(dot_src,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # dot_img = dot_c[0]

    dot_img = cv2.imread('./dot1.jpg', 0)
    dot_img2 = cv2.imread('./dot2.png', 0)
    # _, dot_img = cv2.threshold(dot_img, 27, 255,cv2.THRESH_BINARY)
    # dot_img = cv2.GaussianBlur(cv2.resize(dot_img,(10,10)),(3,3),1)
    #
    # imwrite('./dot1.jpg',dot_img)
    if dot_img is None or dot_img2 is None:
        print('err', '缺失小数点模板')
        sys.exit(0)
    d_w, d_h = dot_img.shape[::-1]
    dots = []
    for i in range(7):
        dots.append(cv2.resize(dot_img, (int(10 / (1 + 0.1 * i)), int(10 / (1 + 0.1 * i)))))
    for i in range(5):
        dots.append(cv2.resize(dot_img, (int(10 * (1 + 0.1 * i)), int(10 * (1 + 0.1 * i)))))
    for i in range(7):
        dots.append(cv2.resize(dot_img2, (int(10 / (1 + 0.1 * i)), int(10 / (1 + 0.1 * i)))))
    for i in range(5):
        dots.append(cv2.resize(dot_img2, (int(10 * (1 + 0.1 * i)), int(10 * (1 + 0.1 * i)))))

    rate = 0.6



    # dot_img = cv2.imread('./dot1.jpg', 0)
    #     d_w, d_h = dot_img.shape[::-1]
    #     dots = []
    #     for i in range(3):
    #         dots.append(cv2.resize(dot_img,(int(10/(1+0.1*i)),int(10/(1+0.1*i)))))
    #     for i in range(2):
    #         dots.append(cv2.resize(dot_img,(int(10*(1+0.1*i)),int(10*(1+0.1*i)))))
    #     rate = 0.6



    print('3.处理数字')
    b_index=0

    result_arr=[]
    cnn = load_cnn()
    for b in blocks_sets:
        b_index += 1
        one_ = train_o[b[1]:b[3], b[0]:b[2]]
        # adapt_one = adapt[b[1]:b[3], b[0]:b[2]]



        imwrite(dir_path+'/3_numblock' + str(b_index) + '.jpg', one_)

        # one_gray = cv2.cvtColor(one_,cv2.COLOR_RGB2GRAY)
        print('亮度提取')
        lightmask,light = get_light_mask(one_, b_index)
        print('二值化')
        adapt = adapt_otsu(one_, b_index)
        print('二值化与亮度提取结果取交集')
        res_block = fillAnd(lightmask, adapt, b_index)




        # ocr_img = Image.fromarray(cv2.cvtColor(res_block, cv2.COLOR_GRAY2RGB))
        # a = image_to_string(ocr_img)
        # print(a,11)

        print('倾斜矫正')
        correct,canny_img = correctAngle(res_block)
        imwrite(dir_path + '/4_canny' + str(b_index) + '.jpg', canny_img)
        imwrite(dir_path + '/4_correct' + str(b_index) + '.jpg', correct)



        print('膨胀数字模板')
        res_block = cv2.erode(correct, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        # res_block = cv2.dilate(res_block, kernel)
        imwrite(dir_path + '/5_erode_and' + str(b_index) + '.jpg', res_block)

        num_mask = makedilateMask(res_block,light)
        imwrite(dir_path + '/5_mask' + str(b_index) + '.jpg', num_mask)
        nums_sets,Line_Dot = cutImage(num_mask)
        if len(nums_sets)==0:
            print('err', '此区域内未检测到数字区域')
            sys.exit(0)

        ind = 0
        line_ind = 0
        for line in nums_sets:
            line_dots = []
            res_list = []
            min_y=np.Inf
            max_y=0
            for [x1,y1,x2,y2] in line:
                if y1<min_y:
                    min_y =y1
                if y2>max_y:
                    max_y=y2

            for o in range(len(line)):
                one_set = line[o]
                if o<len(line)-1:
                    a_num = correct[min_y:max_y, one_set[0]:line[o+1][0]]

                else:
                    a_num = correct[min_y:max_y, one_set[0]:one_set[2]]
                # a_gray = one_gray[one_set[1]:one_set[3], one_set[0]:one_set[2]]
                # imwrite(dir_path+'/9_a_num' + str(ind) + '.jpg', a_num)

                img_width =32
                img_height = 64
                scale_x=a_num.shape[1]/32
                scale_y=a_num.shape[0]/64
                if scale_x > scale_y:
                    scale = scale_x
                    real_w = img_width
                    real_h = int(a_num.shape[0] / scale)
                    black_img = np.zeros((img_height, img_width), np.uint8)
                    img = cv2.resize(a_num, (real_w, real_h))

                    t_ = int((img_height - real_h) / 2)
                    black_img[t_:t_ + real_h, :] = img
                else:
                    scale = scale_y
                    real_w = int(a_num.shape[1] / scale)
                    real_h = img_height
                    black_img = np.zeros((img_height, img_width), np.uint8)
                    img = cv2.resize(a_num, (real_w, real_h))
                    t_ = int((img_width - real_w) / 2)
                    black_img[:, t_:t_ + real_w] = img
                # black_img = cv2.resize(a_num, (32, 64))
                a_num = cv2.GaussianBlur(black_img, (3, 3), 0)
                tmp_dot_max = 0
                x = a_num.astype(float)
                x *= (1. / 255)
                x = np.array([x])
                x = x.reshape(1, 64, 32, 1)
                result = cnn.predict(x)
                print('数字：', convert2Num(result))
                res_list.append(convert2Num(result))
                a_num1 = a_num.copy()
                for i in range(len(dots)):
                    # _, a_num =  cv2.threshold(a_num, 127, 255,cv2.THRESH_BINARY)
                    res = cv2.matchTemplate(a_num, dots[i], cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= rate)
                    # print(np.max(res))
                    # print(loc)
                    for pt in zip(*loc[::-1]):
                        if pt[0] >= 3 and pt[1] >= 32:
                            if res[pt[1]][pt[0]] > tmp_dot_max:
                                tmp_dot_max = res[pt[1]][pt[0]]
                            # cv2.rectangle(a_num1, pt, (pt[0] + d_w, pt[1] + d_h), 255, 1)
                # _, a_num = cv2.threshold(a_num, 127, 255, cv2.THRESH_BINARY)
                # _, cnts, hierarchy = cv2.findContours(a_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # for c in cnts:
                #     tmp = match_dot(dot_img,c)
                #     # print('lunkuo',tmp)
                #     if tmp<tmp_dot_max:
                #         tmp_dot_max = tmp

                line_dots.append(tmp_dot_max)
                imwrite(dir_path+'/6_a_num' + str(b_index) + '_' + str(ind) + '.jpg', a_num1)
                ind += 1
            print('小数点arr:', line_dots)
            dot_ind = np.where(line_dots == np.max(line_dots))[0][0]

            if Line_Dot[line_ind] is not None and line_dots[dot_ind]<0.95:
                if res_list[0]=='-' and Line_Dot[line_ind] ==0:
                    if line_dots[dot_ind] != 0:
                        # print('小数点：',dot_ind)
                        res_list.insert(dot_ind + 1, '.')
                    else:
                        print('区域内小数点检测失败!')
                else:
                    res_list.insert(Line_Dot[line_ind], '.')
            else:
                if line_dots[dot_ind] != 0:
                    # print('小数点：',dot_ind)
                    res_list.insert(dot_ind + 1, '.')
                else:
                    print('区域内小数点检测失败!')

            result = ''.join(res_list)
            print('res', result)
            result_arr.append(result)
            line_ind +=1

    f = open(outPath, 'w')
    f.write('\n' + path)
    for i in range(len(result_arr)):
        f.write(' 结果' + str(i) + '：' + str(result_arr[i]))
    f.close()
    print('结果输出在' + outPath)



def bright_avg(src):
    '''
    计算亮度
    :param src: 灰度图
    :return:
    '''
    avg,stdd = cv2.meanStdDev(src)
    # print(avg)
    return avg[0][0],stdd[0][0]


# if __name__ == '__main__':
#     for i in range(1,37):
#         # pa = './img/front/pic'+str(i)+'.jpg'
#         pa = './img/多角度拍摄/角度2/a2_'+str(i)+'.jpg'
#         # pa = './img/front/dark/d'+str(i)+'.jpg'
#         # pa = './img/front/pic'+str(i)+'.jpg'
#         # main_process(pa)
#         process1(pa,i)



## 问题图片：cv2.imread('./img/多角度拍摄/角度3/a_3.jpg') 数字分割


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
            # img_path = './img2/image20181212上午112250.jpg'
            # for ii in range(209,210):
            #     img_path = './img2/img'+str(ii)+'.jpg'
                # img_path = './img2/no/img'+str(ii)+'.jpg'
                # img_path = './img2/img0.jpg'
                # img_path = './img2/no/image20181212上午101234.jpg'
                # img_path = './img2/no/image20181212上午111909.jpg'
                # img_path = './img2/no/image20181212上午112255.jpg'
                # img_path = './img2/no/image20181212上午112756.jpg'
                # img_path = './img2/no/img179.jpg' #0->9 小数点
                # img_path = './img2/no/im0.jpg' #小数点
            dir_path_arr =img_path.split(os.path.sep)[:-1]
            filename = img_path.split(os.path.sep)[-1].split('.')[0]
            dir_path = os.path.sep.join(dir_path_arr) + os.path.sep + filename
            isExists = os.path.exists(dir_path)
            if not isExists:
                os.makedirs(dir_path)
                # os.makedirs(dir_path+'/num1')
                # os.makedirs(dir_path+'/num2')
            process1(img_path)
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
        process1(img_path, out_path)




