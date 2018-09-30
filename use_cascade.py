import cv2
import numpy as np




kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))



def cutImage(amask, origin):
    _, contours, hierarchy = cv2.findContours(amask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    cont_img = origin.copy()
    for i in range(len(c)):
        rect = cv2.minAreaRect(c[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(cont_img, [box], -1, (0, 0, 255), -1)
    # 绘制结果
    cv2.imwrite('./res_cas/3_1red_contours.jpg', cont_img)
    block_arr = []
    for cur in c:
        numBox = np.int0(cv2.boxPoints(cv2.minAreaRect(cur)))
        Xs = [i[0] for i in numBox]
        Ys = [i[1] for i in numBox]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cut = origin[y1:y1 + hight, x1:x1 + width]
        block_arr.append(cut)
    return block_arr[0]


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
    print(len(matches))
    img3 = cv2.drawMatches(src1, kp1, src2, kp2, matches[:200], None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite('./res_cas/1_match.jpg', img3)

    img_points = np.zeros(src1.shape[:2], np.uint8)
    for point in kp1:
        cv2.circle(img_points, (int(point.pt[0]), int(point.pt[1])), 30, 255, -1)

    # img_points = cv2.morphologyEx(img_points, cv2.MORPH_OPEN, kernel5, iterations=5)
    img_points = cv2.dilate(img_points, kernel2, iterations=4)
    cv2.imwrite('./res_cas/2_points.jpg', img_points)

    return img_points




def pro_resize(src,default=-1):

    (h, w,_) = src.shape
    scale = 300. / w
    if default!=-1:
        scale = default/w
    src = cv2.resize(src, (int(w*scale), int(h*scale)))
    return scale,src


def detective_img(src):
    watch_cascade = cv2.CascadeClassifier('./cascade/cascade64_h.xml')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rows,cols,cla = src.shape
    # t0 = time.time()
    watches = watch_cascade.detectMultiScale(gray, 1.05, 2)
    # print(time.time() - t0)
    tmp = watches[:1]
    for (x, y, w, h) in watches:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 5)
    rect_set = []
    for (x, y, w, h) in tmp:
        x1 = x-20 if x-20>0 else 0
        y1 = y-20 if x-20>0 else 0
        x2 = x + w+20 if x + w+20<cols else cols
        y2 = y + h+20 if y + h+20<rows else rows
        cv2.rectangle(src, (x1, y1), (x2 , y2), (0, 0, 255), 5)
        rect_set=[x1,y1,x2, y2]

    cv2.imwrite('./res_cas/4_detective.jpg',src)
    return rect_set

def detective_img1(src):
    watch_cascade = cv2.CascadeClassifier('./cascade/cascade128.xml')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rows,cols,cla = src.shape
    # t0 = time.time()
    watches = watch_cascade.detectMultiScale(gray, 1.05, 2)
    # print(time.time() - t0)
    tmp = watches[:1]
    for (x, y, w, h) in watches:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 5)
    rect_set = []
    for (x, y, w, h) in tmp:
        x1 = x-30 if x-30>0 else 0
        y1 = y-30 if x-30>0 else 0
        x2 = x + w+30 if x + w+30<cols else cols
        y2 = y + h+30 if y + h+30<rows else rows
        cv2.rectangle(src, (x1, y1), (x2 , y2), (0, 0, 255), 5)
        rect_set=[x1,y1,x2, y2]

    cv2.imwrite('./res_cas/6_detective.jpg',src)
    return rect_set


def get_light_mask(src):
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    mask4_l = cv2.inRange(V, 240, 255)
    cv2.imwrite('./res_cas/5_light.jpg', mask4_l)
    return mask4_l

if __name__ == '__main__':
    query = cv2.imread('./img/query.jpg')

    # query = cv2.cvtColor(cv2.GaussianBlur(query, (7, 7), 1), cv2.COLOR_RGB2GRAY)

    # train_o = cv2.imread('./img/front/pic2.jpg') #待检测图
    # train_o = cv2.imread('./img/多角度拍摄/角度2/a2_8.jpg')
    train_o = cv2.imread("./img/front/dark/d7.jpg")
    # train_o = cv2.imread('./img/多角度拍摄/角度4/aa11.jpg')
    # train_o = cv2.imread('./img/im3.jpg')
    # train_o = cv2.imread('./img/多角度拍摄/角度1/a1_15.jpg')
    # train_o=cv2.imread('./img/front/pic5.jpg')

    train = train_o
    # train = cv2.cvtColor(cv2.GaussianBlur(train_o, (7, 7), 1), cv2.COLOR_RGB2GRAY)

    im_zeros = matchArea(train,query)

    # train = cv2.cvtColor(cv2.GaussianBlur(train_o, (7, 7), 1), cv2.COLOR_RGB2GRAY)
    instru_area = cutImage(im_zeros,train)


    scale_rate,mini_instru = pro_resize(instru_area)

    rect = detective_img(mini_instru)
    rect = [int(ii/scale_rate) for ii in rect]

    num_orig = instru_area[rect[1]:rect[3],rect[0]:rect[2]]

    # scale2,num_orig = pro_resize(cv2.GaussianBlur(num_orig,(7,7),1),500.)

    train_gray = cv2.cvtColor(num_orig,cv2.COLOR_RGB2GRAY)
    # kernel = np.ones((5, 5), np.float32) / 25
    # train_gray = cv2.filter2D(train_gray, -1, kernel)

    # num_canny = cv2.Canny(train_gray, 20, 300)
    bina1  = cv2.adaptiveThreshold(train_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
    threshold, bina2 = cv2.threshold(train_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    cv2.imwrite('./res_cas/5_adape.jpg', bina1)
    # num_bina = bina[rect[1]:rect[3],rect[0]:rect[2]]
    # cv2.imwrite('./res_cas/5_numarea.jpg', num_bina)



