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



def imwrite(_path,_img):
    cv2.imencode('.jpg',_img)[1].tofile(_path)
    print('img',_path)



def calc_gamma(src,rate):
    img0 = src


    def gamma_trans(img, gamma):
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 实现映射用的是Opencv的查表函数
        return cv2.LUT(img0, gamma_table)

    img0_corrted = gamma_trans(img0, rate)

    imwrite(dir_path + '/0_gamma.jpg', img0_corrted)
    return img0_corrted



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
    for box in y_pred_decoded[0]:
        xmin = int(box[2]-5) if int(box[2]-5)>=0 else 0
        ymin = int(box[3])
        xmax = int(box[4] + 5)
        ymax = int(box[5] + 5 )
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
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
    K.clear_session()
    return mask,all_blocks


def bright_avg(src):
    '''
    计算亮度
    :param src: 灰度图
    :return:
    '''
    avg,stdd = cv2.meanStdDev(src)
    # print(avg)
    return avg[0][0],stdd[0][0]

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

    all_light,stdd = bright_avg(train_o)
    print('gamma校正前的平均亮度',all_light)
    if all_light<70:
        rate=0.8-(60-all_light)*0.01
    elif all_light>=90:
        rate = 2.2+(all_light-90)*0.01
    else:
        rate = 1.5
    print('gamma数值',rate)
    train_o = calc_gamma(train_o,rate)

    all_light, stdd = bright_avg(train_o)
    print('gamma校正后的平均亮度', all_light)
    #
    # if light>80:
    # light_mask = split_light(train_o,index)
    # print('1.获得数字区域')
    area_mask, blocks_sets = predict_num_area(train_o)





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
            # for ii in range(117,118):
            # img_path = './img2/img'+str(ii)+'.jpg'
            # img_path = './img2/no/im'+str(ii)+'.jpg'
            # img_path = './img2/img0.jpg'
            # img_path = './img2/no/image20181212上午101234.jpg'
            # img_path = './img2/no/image20181212上午111909.jpg'
            # img_path = './img2/no/image20181212上午112255.jpg'
            # img_path = './img2/no/image20181212上午112756.jpg'
            img_path = './img2/no/im32.jpg' #0->9 小数点
            # img_path = './img2/no/im1.jpg' #0->9 小数点
            # img_path = './img2/no/im16.jpg' #0->9 小数点
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


