import cv2
import numpy as np




if __name__ == '__main__':
    img1 = cv2.imread('./img/cam2.png')
    img1 = cv2.bilateralFilter(img1, 9, 70,70)
    img_hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)  # 分离 HSV 三通道
    Lowerred0 = np.array([155, 43, 46])
    Upperred0 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, Lowerred0, Upperred0)
    Lowerred1 = np.array([0, 43, 46])
    Upperred1 = np.array([11, 255, 255])
    mask2 = cv2.inRange(img_hsv, Lowerred1, Upperred1)  # 将红色区域部分归为全白，其他区域归为全黑
    Lowerred2 = np.array([35, 43, 46])
    Upperred2 = np.array([77, 255, 255])
    mask3 = cv2.inRange(img_hsv, Lowerred2, Upperred2)
    img_hsv = mask1 + mask2 + mask3

    cv2.imwrite('./hsv/img_hsv.jpg',img_hsv)
    mask4 = cv2.inRange(V, 200, 255)
    cv2.imwrite('./hsv/img_hsv_.jpg', mask4)


    img1 = cv2.imread('./img/多角度拍摄/角度1/a1_17.jpg')
    img1 = cv2.bilateralFilter(img1, 9, 70,70)
    img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)  # 分离 HSV 三通道
    Lowerred0 = np.array([155, 43, 46])
    Upperred0 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, Lowerred0, Upperred0)
    Lowerred1 = np.array([0, 43, 46])
    Upperred1 = np.array([11, 255, 255])
    mask2 = cv2.inRange(img_hsv, Lowerred1, Upperred1)  # 将红色区域部分归为全白，其他区域归为全黑
    Lowerred2 = np.array([35, 43, 46])
    Upperred2 = np.array([77, 255, 255])
    mask3 = cv2.inRange(img_hsv, Lowerred2, Upperred2)
    img_hsv = mask1 + mask2 + mask3

    cv2.imwrite('./hsv/img_hsv1.jpg', img_hsv)
    mask4 = cv2.inRange(V,250,255)
    cv2.imwrite('./hsv/img_hsv11.jpg', mask4)


    img1 = cv2.imread('./img/front/pic5.jpg')
    img1 =cv2.bilateralFilter(img1, 9, 70,70)
    img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)  # 分离 HSV 三通道
    Lowerred0 = np.array([155, 43, 46])
    Upperred0 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, Lowerred0, Upperred0)
    Lowerred1 = np.array([0, 43, 46])
    Upperred1 = np.array([11, 255, 255])
    mask2 = cv2.inRange(img_hsv, Lowerred1, Upperred1)  # 将红色区域部分归为全白，其他区域归为全黑
    Lowerred2 = np.array([35, 43, 46])
    Upperred2 = np.array([77, 255, 255])
    mask3 = cv2.inRange(img_hsv, Lowerred2, Upperred2)
    img_hsv = mask1 + mask2 +mask3

    cv2.imwrite('./hsv/img_hsv2.jpg', img_hsv)
    mask4 = cv2.inRange(V, 250, 255)
    cv2.imwrite('./hsv/img_hsv22.jpg', mask4)



    img1_o = cv2.imread("./img/im2.jpg")
    img1 = cv2.GaussianBlur(img1_o,(5,5),0)
    # img1 =  cv2.bilateralFilter(img1_o, 9, 70,70)
    img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img_hsv)  # 分离 HSV 三通道
    Lowerred0 = np.array([125, 43, 46])
    Upperred0 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, Lowerred0, Upperred0)
    Lowerred1 = np.array([0, 43, 46])
    Upperred1 = np.array([11, 255, 255])
    mask2 = cv2.inRange(img_hsv, Lowerred1, Upperred1)  # 将红色区域部分归为全白，其他区域归为全黑
    Lowerred2 = np.array([35, 43, 46])
    Upperred2 = np.array([77, 255, 255])
    mask3 = cv2.inRange(img_hsv, Lowerred2, Upperred2)
    img_hsv = mask1 + mask2 + mask3
    cv2.imwrite('./hsv/img_hsv3.jpg', img_hsv)

    mask4 = cv2.inRange(V, 250, 255)
    cv2.imwrite('./hsv/img_hsv33.jpg', mask4)
    output = cv2.bitwise_and(img1_o, img1_o, mask=mask4)
    cv2.imwrite('./hsv/img_hsv3_out.jpg', output)

    _, contours, hierarchy = cv2.findContours(img_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    cont_img = img1_o.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), 2)
    # 绘制结果
    cv2.imwrite('./hsv/red_contours.jpg', cont_img)