import os
import cv2


img_dir = '/Users/weng/Documents/大四/NumInstrument/img2'

out_dir = '/Users/weng/Documents/大四/NumInstrument/model2/generate_imgs/formal'


def copy_img():
    index=0
    for i in range(0,265):
        index+=1
        folder_path = img_dir+'/img'+str(i)
        if os.path.exists(folder_path):
            for j in range(1,4):
                img1 =  cv2.imread(folder_path+'/3_erode_and'+str(j)+'.jpg')
                img2 =  cv2.imread(folder_path+'/3_light'+str(j)+'.jpg')
                img3 =  cv2.imread(folder_path+'/4_correct'+str(j)+'.jpg')
                if img1 is not None:
                    cv2.imwrite(out_dir+'/image'+str(index)+'_1.jpg',img1)
                if img2 is not None:
                    cv2.imwrite(out_dir+'/image'+str(index)+'_2.jpg',img2)
                if img3 is not None:
                    cv2.imwrite(out_dir+'/image'+str(index)+'_3.jpg',img3)


if __name__ == '__main__':
    copy_img()