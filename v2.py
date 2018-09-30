#使用sift特征匹配两幅图：

import cv2
import numpy as np




# BFmatch(暴力匹配)：计算匹配图层的一个特征描述子与待匹配图层的所有特征描述子的距离返回最近距离。
# 上代码：
query=cv2.imread('./img/front/pic3.jpg')

query = cv2.cvtColor(cv2.GaussianBlur(query, (5, 5), 0) , cv2.COLOR_RGB2GRAY)

train=cv2.imread('./img/多角度拍摄/角度2/IMG_1795.jpg')
# train=cv2.imread('./img/front/pic5.jpg')
# train=cv2.imread("./img/front/dark/d7.jpg")
train = cv2.cvtColor(cv2.GaussianBlur(train, (5, 5), 0) , cv2.COLOR_RGB2GRAY)


# 暴力匹配
orb=cv2.ORB_create()
kp1,des1=orb.detectAndCompute(train,None)
kp2,des2=orb.detectAndCompute(query,None)
# 针对ORB算法 NORM_HAMMING 计算特征距离 True判断交叉验证
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
# 特征描述子匹配
matches=bf.match(des1,des2)


matches=sorted(matches,key=lambda x:x.distance)
print('kp1: ',kp1)
print('kp2: ',kp2)
print(len(matches))

im_zeros = np.zeros(query.shape,np.uint8)

for point in kp1:
    cv2.circle(im_zeros,(int(point.pt[0]),int(point.pt[1])), 5, (255, 255, 255), -1)

cv2.imwrite('./v2/points.jpg',im_zeros)


img3=cv2.drawMatches(train,kp1,query,kp2,matches[:100],None,matchColor = (0,255,0),flags=2)
cv2.imwrite('./v2/match.jpg',img3)







# # FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
# # 上代码：
# query=cv2.imread('./img/front/pic3.jpg')
# query = cv2.cvtColor(cv2.GaussianBlur(query, (5, 5), 0) , cv2.COLOR_RGB2GRAY)
#
# train=cv2.imread('./img/多角度拍摄/角度2/IMG_1795.jpg')
# # train=cv2.imread("./img/front/dark/d7.jpg")
# train = cv2.cvtColor(cv2.GaussianBlur(train, (5, 5), 0) , cv2.COLOR_RGB2GRAY)
#
#
# sift=cv2.xfeatures2d.SIFT_create()
# kp1,des1=sift.detectAndCompute(train,None)
# kp2,des2=sift.detectAndCompute(query,None)
#
#
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
#
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
#
#
# matches = flann.knnMatch(des1,des2,k=2)
#
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
#
#
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#
# #如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
#
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
#
#
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
#
#
# img3 = cv2.drawMatchesKnn(train,kp1,query,kp2,matches,None,**draw_params)
# cv2.imwrite('./v2/match.jpg',img3)