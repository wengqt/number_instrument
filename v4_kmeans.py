

import numpy as np
import PIL.Image as Image
import cv2
from sklearn.cluster import KMeans



kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))

def load_data(img,step=1):
    # f = open(file_path,'rb') #二进制打开
    data = []
    # img = Image.open(f) #以列表形式返回图片像素值
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    m,n = img.size #活的图片大小
    print(m,n)
    if step==1:
        img = img.resize((int(m/2),int(n/2)))
        m, n = img.size  # 活的图片大小
    for i in range(m):
        for j in range(n):  #将每个像素点RGB颜色处理到0-1范围内并存放data
            # print(img.getpixel((i,j)))
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    # f.close()
    print('data: ', data[0])
    return data,m,n,img

    # img = Image.open(file_path)
    # # img = img.convert('L')
    # m, n = img.size  # 活的图片大小
    # # print(m,n)
    # img = img.resize((int(m/4),int(n/4)))
    # hight, width = img.size
    # img = np.asarray(img, dtype='float64') / 256.
    # # print(img)
    # # tmp = img.reshape(-1,1)
    # tmp = img.reshape([-1,3])
    #
    # print('tmp: ',tmp)
    # return tmp,width,hight #以矩阵型式返回data，图片大小


def gray_cluster(img,step=1):
    img_data,row,col,smallorigin = load_data(img,step)
    if step==2:
        label = KMeans(n_clusters=3).fit_predict(img_data)  #聚类中心的个数为5
        label = label.reshape((row,col))    #聚类获得每个像素所属的类别
        pic_new = Image.new("L",(row,col))
        for i in range(row):    #根据所属类别向图片中添加灰度值
            for j in range(col):
                if label[i][j] == 2:
                    pic_new.putpixel((i,j),256)
        pic_new.save('./v3/222.jpg')
        pic_new = np.asarray(pic_new)
    else:
        label = KMeans(n_clusters=5).fit_predict(img_data)  # 聚类中心的个数为5
        label = label.reshape((row, col))  # 聚类获得每个像素所属的类别
        pic_new = Image.new("L", (row, col))
        for i in range(row):  # 根据所属类别向图片中添加灰度值
            for j in range(col):
                if label[i][j] == 4:
                    pic_new.putpixel((i, j), 256)
        pic_new.save('./v3/444.jpg')
        pic_new = np.asarray(pic_new)

    return pic_new,cv2.cvtColor(np.asarray(smallorigin),cv2.COLOR_RGB2BGR)

# pic_new = Image.new("L",(row,col))  #创建一张新的灰度图保存聚类后的结果
# print( pic_new.size)
# for i in range(row):    #根据所属类别向图片中添加灰度值
#     for j in range(col):
#         if label[i][j] == 0:
#             pic_new.putpixel((i,j),int(256/(1)))
# pic_new.save('./v3/000.jpg')
# pic_new = Image.new("L",(row,col))
# for i in range(row):    #根据所属类别向图片中添加灰度值
#     for j in range(col):
#         if label[i][j] == 1:
#             pic_new.putpixel((i,j),int(256/(1)))
# pic_new.save('./v3/111.jpg')
# pic_new = Image.new("L",(row,col))
# for i in range(row):    #根据所属类别向图片中添加灰度值
#     for j in range(col):
#         if label[i][j] == 2:
#             pic_new.putpixel((i,j),int(256/(1)))
# pic_new.save('./v3/222.jpg')
# pic_new = Image.new("L",(row,col))
# for i in range(row):    #根据所属类别向图片中添加灰度值
#     for j in range(col):
#         if label[i][j] == 3:
#             pic_new.putpixel((i,j),int(256/(1)))
# pic_new.save('./v3/333.jpg')

# pic_new = Image.new("L",(row,col))
# for i in range(row):    #根据所属类别向图片中添加灰度值
#     for j in range(col):
#         if label[i][j] == 5:
#             pic_new.putpixel((i,j),int(256/(1)))
# pic_new.save('./v3/555.jpg')



