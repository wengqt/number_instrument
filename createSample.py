
import cv2


for i in range(0,90):
    img = cv2.imread('../train_need_resize/img'+str(i)+'.png')
    img = cv2.resize(img,(128,64))
    cv2.imwrite('./get_train_area/img'+str(i)+'.jpg',img)