import cv2


def resize_dot(img_path,out_path):
    img = cv2.imread(img_path,0)

    img = cv2.resize(img,(10,10))

    # img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)))
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(out_path,img)


resize_dot('./dot7.png','./dot2.jpg')