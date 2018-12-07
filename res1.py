'''
这个版本是不使用其他图片，直接识别的版本。
使用ssd模型版本
'''




from keras import backend as K
import numpy as np
from keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import cv2
import os
import sys
# from imageio import imread
# import peakdetective


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




def cutImage(amask, origin,model=1):
    '''

    :param amask:
    :param origin:
    :param model: model!=1 return all the rectangles' [x1,y1,x2,y2]
    :return:
    '''
    _, contours, hierarchy = cv2.findContours(amask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def matchArea(src1,src2):
    src1 = cv2.cvtColor(cv2.GaussianBlur(src1, (7, 7), 1), cv2.COLOR_RGB2GRAY)
    src2 = cv2.cvtColor(cv2.GaussianBlur(src2, (7, 7), 1), cv2.COLOR_RGB2GRAY)
    # 暴力匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(src1, None)
    kp2, des2 = orb.detectAndCompute(src2, None)
    # 针对ORB算法 NORM_HAMMING 计算特征距离 True判断交叉验证
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 特征描述子匹配
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    # print(len(matches))
    img3 = cv2.drawMatches(src1, kp1, src2, kp2, matches[:200], None, matchColor=(0, 255, 0), flags=2)
    imwrite(dir_path+'/1_match.jpg', img3)

    img_points = np.zeros(src1.shape[:2], np.uint8)
    for point in kp1:
        cv2.circle(img_points, (int(point.pt[0]), int(point.pt[1])), 30, 255, -1)

    # img_points = cv2.morphologyEx(img_points, cv2.MORPH_OPEN, kernel5, iterations=5)
    img_points = cv2.dilate(img_points, getKernel_2(src1.shape[1]), iterations=4)
    imwrite(dir_path+'/2_points.jpg', img_points)

    return img_points

def predict_num_area(src):
    '''
    :param src:opencv读取的图
    :return: scale:缩放倍率，mask:数字区域的全白图,all_blocks:矩形坐标
    '''

    model_path = 'ssd7_pascal_07_epoch-17_loss-0.8387_val_loss-0.8608.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session()  # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'compute_loss': ssd_loss.compute_loss})
    img_height = 341
    img_width = 256
    scale_y = src.shape[0]/341
    scale_x = src.shape[1]/256
    normalize_coords = True
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.

    # We'll only load one image in this example.
    # filename = '../NumInstrument/img/im3.JPG'
    # filename='../ssd_trains/JPEGImages/image1024.JPG'

    # img = cv2.imread(filename)
    img = cv2.resize(src, (img_width, img_height))
    orig_images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    mask = np.zeros((341,256),np.uint8)
    for box in y_pred_decoded[0]:
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4] + 5)
        ymax = int(box[5] + 5 )
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
        cv2.rectangle(input_images[0], (xmin, ymin), (xmax, ymax), 255, 2)

    input_images[0] = cv2.cvtColor(input_images[0], cv2.COLOR_RGB2BGR)


    # mask = cv2.dilate(mask, kernel4)

    imwrite(dir_path+'/4_pre_mask.jpg', mask)
    # imwrite(dir_path+'/4_pre_res' + str(index) + '.jpg', input_images[0])

    all_blocks = cutImage(mask,input_images[0],-1)
    # print(all_blocks,scale)
    if len(all_blocks)==0:
        print('err', '未找到数字区域')
        sys.exit(0)

    tmp=[]
    for abox in all_blocks:
        aa=[int(abox[0]*scale_x),int(abox[1]*scale_y),int(abox[2]*scale_x),int(abox[3]*scale_y)]
        tmp.append(aa)
    all_blocks = tmp
    K.clear_session()
    return mask,all_blocks

def get_light_mask(src,index):
    # src = calc_equalize(src)
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)
    # print('亮度均值：',np.mean(V))

    low=200
    if np.mean(V) >=100:
        low=240
    if p1!=-1:
        low = p1
    print('当前亮度提取低阈值为' + str(low))
    mask4_l = cv2.inRange(V, low, 255)
    imwrite(dir_path+'/3_light'+str(index)+'.jpg', mask4_l)

    return mask4_l

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

def correctAngle(src):
    origin_src = src.copy()
    if src.shape[1]<300:
        default_v = 80
    else:
        default_v =150
    canny = cv2.Canny(cv2.GaussianBlur(src, (5, 5), 0), 0, default_v)
    imwrite(dir_path+'/4_canny.jpg', canny)
    canny = cv2.dilate(canny, kernel3, iterations=1)
    # imwrite(dir_path+'/4_cannyDilate.jpg', canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 50)
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

    averageAngle = (angle / float(numLine)) * 180 / np.pi
    print('averageAngle : %f' % averageAngle)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), averageAngle - 90, 1.0)

    correct = cv2.warpAffine(origin_src, M, (width, height))

    imwrite(dir_path+'/7_correct.jpg', correct)
    return correct








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


def makedilateMask(src):
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

    mask = cv2.dilate(src, kel)
    imwrite(dir_path+'/5_mask.jpg',mask)

    return mask


def load_cnn():
    model_path = './myCNN.h5'
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
        low=250
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
    train_o = cv2.imread(path)
    if train_o is None:
        print('err', '找不到图片'+path)
        sys.exit(0)
    # train_o = cv2.imread("./img/front/dark/d9.jpg")
    # train_o = cv2.imread("./img/front/pic20.jpg")

    # train_gray =cv2.cvtColor(train_o,cv2.COLOR_BGR2GRAY)


    #
    # if light>80:
    # light_mask = split_light(train_o,index)
    print('1.获得数字区域')
    area_mask, blocks_sets = predict_num_area(train_o)

    # adapt = adapt_otsu(train_o, index)
    print('2.加载小数点模板')
    dot_src = cv2.imread('./dot.jpg', 0)
    if dot_src is None:
        print('err','缺失小数点模板')
        sys.exit(0)
    _, dot_src = cv2.threshold(dot_src, 127, 255,cv2.THRESH_BINARY)
    _, dot_c, hierarchy = cv2.findContours(dot_src,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_img = dot_c[0]
    # cv2.drawContours(dot_src,[dot_img],-1,255)
    # imwrite('./dot.jpg',dot_src)

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
    num_blocks=[]
    result_arr=[]
    cnn = load_cnn()
    for b in blocks_sets:
        b_index += 1
        one_ = train_o[b[1]:b[3], b[0]:b[2]]
        # adapt_one = adapt[b[1]:b[3], b[0]:b[2]]
        imwrite(dir_path+'/3_numblock' + str(b_index) + '.jpg', one_)
        print('亮度提取')
        lightmask = get_light_mask(one_, b_index)
        print('二值化')
        adapt = adapt_otsu(one_, b_index)
        print('二值化与亮度提取结果取交集')
        res_block = fillAnd(lightmask, adapt, b_index)
        kernel = getKernel(res_block.shape[1])
        res_block = cv2.erode(res_block, kernel)
        imwrite(dir_path+'3_erode_and' + str(b_index) + '.jpg', res_block)
        num_blocks.append(res_block)
        print('倾斜矫正')
        correct = correctAngle(res_block)
        print('膨胀数字模板')
        num_mask = makedilateMask(correct)
        nums_sets = cutImage(num_mask, correct, 2)
        if len(nums_sets)==0:
            print('err', '此区域内未检测到数字区域')
            sys.exit(0)

        ind = 0
        for line in nums_sets:
            line_dots = []
            res_list = []
            for one_set in line:
                a_num = correct[one_set[1]:one_set[3], one_set[0]:one_set[2]]
                # imwrite(dir_path+'/9_a_num' + str(ind) + '.jpg', a_num)
                # a_num = cv2.resize(a_num, (32, 64))
                a_num = cv2.GaussianBlur(cv2.resize(a_num, (32, 64)), (3, 3), 1)
                tmp_dot_max = np.Inf
                x = a_num.astype(float)
                x *= (1. / 255)
                x = np.array([x])
                x = x.reshape(1, 64, 32, 1)
                result = cnn.predict(x)
                print('数字：', convert2Num(result))
                res_list.append(convert2Num(result))
                # for i in range(5):
                #     res = cv2.matchTemplate(a_num, dots[i], cv2.TM_CCOEFF_NORMED)
                #     loc = np.where(res >= rate)
                #     # print(np.max(res))
                #     # print(loc)
                #     for pt in zip(*loc[::-1]):
                #         if pt[0] >= 16 and pt[1] >= 45:
                #             if res[pt[1]][pt[0]] > tmp_dot_max:
                #                 tmp_dot_max = res[pt[1]][pt[0]]
                #             cv2.rectangle(a_num, pt, (pt[0] + d_w, pt[1] + d_h), 255, 1)
                _, a_num = cv2.threshold(a_num, 127, 255, cv2.THRESH_BINARY)
                _, cnts, hierarchy = cv2.findContours(a_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    tmp = match_dot(dot_img,c)
                    # print('lunkuo',tmp)
                    if tmp<tmp_dot_max:
                        tmp_dot_max = tmp

                line_dots.append(tmp_dot_max)
                imwrite(dir_path+'/6_a_num' + str(ind) + '.jpg', a_num)
                ind += 1
            print('小数点arr:', line_dots)
            dot_ind = np.where(line_dots == np.min(line_dots))[0][0]
            if line_dots[dot_ind] != 0:
                # print('小数点：',dot_ind)
                res_list.insert(dot_ind + 1, '.')
            else:
                print('区域内小数点检测失败!')

            result = ''.join(res_list)
            print('res', result)
            result_arr.append(result)

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
    print(avg)
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




