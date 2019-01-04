import cv2
import random








def img_rotate(src,ind):
    img = cv2.imread(src)
    rows, cols, channel = img.shape
    a= random.random()
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), int(30*a), 1.0)
    print(a)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite('./new/im2'+str(ind)+'.jpg',dst)




def up_down(src,ind):
    img = cv2.imread(src)
    v_flip = cv2.flip(img, 0)
    cv2.imwrite('./new/im0' + str(ind) + '.jpg', v_flip)


def resize(src,ind):
    img = cv2.imread(src)
    rows, cols, channel = img.shape
    if cols>100 and cols<200:
        img = cv2.resize(img,(int(cols/2),int(rows/2)))
    elif cols>200 and cols<300:
        img = cv2.resize(img, (int(cols / 3), int(rows / 3)))
    elif cols>300 and cols<400:
        img = cv2.resize(img, (int(cols / 4), int(rows / 4)))
    elif cols>400 :
        img = cv2.resize(img, (int(cols / 5), int(rows / 5)))
    cv2.imwrite('./new/neg' + str(ind) + '.jpg', img)



def resize_train(src,ind):
    img = cv2.imread(src)
    img = cv2.resize(img,(64,32))
    cv2.imwrite('../JPEGImages/image' + str(ind) + '.jpg', img)



def resize_ori(src,ind):
    img = cv2.imread(src)
    sh=img.shape
    # h = int(sh[0]*256/sh[1])
    img = cv2.resize(img, (256, 341))
    cv2.imwrite('./new/image' + str(ind) + '.jpg', img)



from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from glob import glob
import cv2
import numpy as np

from PIL import Image


def bright_avg(src):
    '''
    计算亮度
    :param src: 灰度图
    :return:
    '''
    avg,stdd = cv2.meanStdDev(src)
    # print(avg)
    return avg[0][0],stdd[0][0]
# 图片生成器
def generator_img(src):
    datagen = ImageDataGenerator(
                rotation_range=5,
                width_shift_range=0.2,
                height_shift_range=0.1,
                rescale=1./255,
                # shear_range=0.05,
                zoom_range=[1.0,1.5],
                # horizontal_flip=True,
                # fill_mode='constant'
                fill_mode='nearest'
    )

    # 打印转换前的图片
    try:
        img = load_img(src)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (456, 256))
        # all_light, stdd = bright_avg(img)
        # print('gamma校正前的平均亮度', all_light)
        # if all_light < 70:
        #     rate = 0.8 - (60 - all_light) * 0.01
        # elif all_light >= 90:
        #     rate = 2.2 + (all_light - 90) * 0.01
        # else:
        #     rate = 1.5
        # print('gamma数值', rate)
        # img = calc_gamma(img, rate)
        #
        # all_light, stdd = bright_avg(img)
        # print('gamma校正后的平均亮度', all_light)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    # img = img.resize((256,144))
    except:
        print('no img '+str(src))
        return 0

    # 将图片转换为数组，并重新设定形状
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # x的形状重组为(1,width,height,channels)，第一个参数为batch_size

    # 这里人工设置停止生成， 并保存图片用于可视化
    i = 0
    for batch in datagen.flow(x,batch_size=1,save_to_dir='./ssd_train/group2',save_prefix='d',save_format='jpg'):
        i +=1
        if i > 2:
            return



def calc_gamma(src,rate):
    img0 = src


    def gamma_trans(img, gamma):
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 实现映射用的是Opencv的查表函数
        return cv2.LUT(img0, gamma_table)

    img0_corrted = gamma_trans(img0, rate)

    # imwrite(dir_path + '/0_gamma.jpg', img0_corrted)
    return img0_corrted





def resize_img(src,ind):
    img = cv2.imread(src)
    if img is None:
        return 0
    img = cv2.resize(img,(456,256))
    cv2.imwrite('./ssd_train/group1_expand/failed256/img' + str(ind) + '.jpg',img)





for i in range(0,320):
    # path='./ssd_train/pic_group1/img'+str(i)+'.jpg'
    path='./img2/img'+str(i)+'.jpg'
    generator_img(path)


# for i in range(0,1014):
#     path = './ssd_train/group1_expand/failed/image' + str(i) + '.jpg'
#     resize_img(path,i)


# generator_img('./img2/img138.jpg')
# for i in range(0,8):
#     # path ='../ssd_trains/JPEGImages/image'+str(i)+'.jpg'
#     # path ='./new/image'+str(i)+'.png'
#     # # generator_img(path)
#     # img = cv2.imread(path)
#     # cv2.imwrite('./new/1/image'+str(i)+'.jpg',img)
#     path = '../train_/-1/a_'+str(i)+'.jpg'
#     generator_img(path)


# for j in range(-1,10):
#     for i in range(0,1140):
#         path1 = '../train_/' + str(j) + '/_' + str(i) + '.jpg'
#         img = cv2.imread(path1)
#         img = cv2.resize(img,(32,64))
#         path2 = '../cnn_train/' + str(j) + '/' + str(i) + '.jpg'
#         cv2.imwrite(path2,img)