
import cv2
import numpy as np

kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# for i in range(0,10):
#     for j in range(0,10):
#         im = cv2.imread('./getArea3/train/'+str(i)+'/'+str(i)+'-'+str(j)+'.jpg')
#         # im = cv2.imread('./getArea3/train/0/0-0.jpg')
#         imBlur = cv2.GaussianBlur(im, (5, 5), 0)
#         imGray = cv2.cvtColor(imBlur, cv2.COLOR_BGR2GRAY)
#         canny = cv2.Canny(imGray, 50, 250)
#         canny =cv2.dilate(canny,kernel3,iterations=1)
#         # _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#         # rect = cv2.minAreaRect(contours[0])
#         h_1, w_1 = canny.shape[:2]
#         mask = np.zeros([h_1 + 2, w_1 + 2], np.uint8)
#         cv2.floodFill(canny,mask,(0, 0), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
#         canny = cv2.bitwise_not(canny)
#         # print(hierarchy[0])
#         # for i in range(0,len(contours)):
#         #     if hierarchy[0][i][3] != -1 :
#         #         if hierarchy[0][hierarchy[0][i][3]][3] == -1:
#         #             cv2.drawContours(canny, contours, i, (255, 255, 255), -1)
#         #         else:
#         #             cv2.drawContours(canny, contours, i, (0, 0, 0), -1)
#         #     else:
#         #         cv2.drawContours(canny, contours, i, (0, 0, 0), -1)
#
#         # cv2.drawContours(canny, contours, -1, (255, 255, 255), -1)
#         canny = cv2.resize(canny, (128, 256))
#         cv2.imwrite('./train/'+str(i)+'/'+str(i)+'_'+str(j)+'.jpg',canny)

i=6
if i ==6:
    for j in range(0,10):
        im = cv2.imread('./getArea3/train/'+str(i)+'/'+str(i)+'-'+str(j)+'.jpg')
        # im = cv2.imread('./getArea3/train/0/0-0.jpg')
        imBlur = cv2.GaussianBlur(im, (5, 5), 0)
        imGray = cv2.cvtColor(imBlur, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(imGray, 50, 250)
        canny =cv2.dilate(canny,kernel3,iterations=1)
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))

        # rect = cv2.minAreaRect(contours[0])
        # h_1, w_1 = canny.shape[:2]
        # mask = np.zeros([h_1 + 2, w_1 + 2], np.uint8)
        # cv2.floodFill(canny,mask,(0, 0), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
        # canny = cv2.bitwise_not(canny)
        # print(hierarchy[0])
        # for i in range(0,len(contours)):
        #     if hierarchy[0][i][3] != -1 :
        #         if hierarchy[0][hierarchy[0][i][3]][3] == -1:
        #             cv2.drawContours(canny, contours, i, (255, 255, 255), -1)
        #         else:
        #             cv2.drawContours(canny, contours, i, (0, 0, 0), -1)
        #     else:
        #         cv2.drawContours(canny, contours, i, (0, 0, 0), -1)
        if j==3 or j==9 :
            cv2.drawContours(canny, contours, 0, (255, 255, 255), -1)
            cv2.drawContours(canny, contours, 2, (255, 255, 255), -1)
            cv2.drawContours(canny, contours, 0, (255, 255, 255), -1)
            cv2.drawContours(canny, contours, 3, (0, 0, 0), -1)
            # cv2.drawContours(canny, contours, 3, (0, 0, 0), -1)
            # cv2.drawContours(canny, contours, 1, (255, 255, 255), -1)
            # cv2.drawContours(canny, contours, 0, (255, 255, 255), -1)
            # cv2.drawContours(canny, contours, 0, (0, 0, 0), -1)
            # cv2.drawContours(canny, contours, 2, (0, 0, 0), -1)
        else:
            cv2.drawContours(canny, contours, 1, (255, 255, 255), -1)
            cv2.drawContours(canny, contours, 2, (0, 0, 0), -1)
            cv2.drawContours(canny, contours, 3, (0, 0, 0), -1)
            # cv2.drawContours(canny, contours, 4, (0, 0, 0), -1)
        canny = cv2.resize(canny, (128, 256))
        cv2.imwrite('./train/' + str(i) + '/' + str(i) + '_' + str(j) + '.jpg', canny)