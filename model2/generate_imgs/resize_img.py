
import cv2
import numpy as np

img_width = 192
img_height = 192


img_dir= '/Users/weng/Documents/大四/NumInstrument/model2/generate_imgs/formal'
out_dir='/Users/weng/Documents/大四/NumInstrument/model2/generate_imgs/out'

def resize_img():

    for i in range(0,660):
        img = cv2.imread(img_dir+'/pic'+str(i)+'.jpg',0)
        (h,w) = img.shape
        scale1 = h/img_height
        scale2 = w/img_width
        if scale1 >scale2:
            scale = scale1
        else:
            scale = scale2

        img = cv2.resize(img,(int(w/scale),int(h/scale)))

        bg = np.zeros((img_height, img_width),np.uint8)

        diff1 = int((img_width-(w/scale))/2)
        diff2 = int((img_height-(h/scale))/2)

        bg[diff2 : diff2+ int(h/scale),diff1 : diff1 + int(w/scale) ] = img
        cv2.imwrite(out_dir+'/pic'+str(i)+'.jpg',bg)



if __name__ == '__main__':
    resize_img()