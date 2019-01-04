
#
# import xml.dom.minidom
#
#
# xml_path = '/Users/weng/Documents/大四/ssd_trains/Annotations'
#
# xmlfile = xml_path+'/image1的副本.xml'
# DomTree = xml.dom.minidom.parse(xmlfile)
# annotation = DomTree.documentElement
# objectlist = annotation.getElementsByTagName('object')
#
# for objects in objectlist:
#     # print objects
#
#     namelist = objects.getElementsByTagName('name')
#     # print 'namelist:',namelist
#     objectname = namelist[0].childNodes[0].data
#     print(objectname)
#
#     bndbox = objects.getElementsByTagName('bndbox')
#     for box in bndbox:
#
#         x1_list = box.getElementsByTagName('xmin')
#         x1 = int(x1_list[0].childNodes[0].data)
#         y1_list = box.getElementsByTagName('ymin')
#         y1 = int(y1_list[0].childNodes[0].data)
#         x2_list = box.getElementsByTagName('xmax')
#         x2 = int(x2_list[0].childNodes[0].data)
#         y2_list = box.getElementsByTagName('ymax')
#         y2 = int(y2_list[0].childNodes[0].data)
#         w = x2 - x1
#         h = y2 - y1
#         y2_list[0].childNodes[0].replaceData(0,1,1)
#
#         print( box.getAttributeNode('xmax'))
#

from xml.etree.ElementTree import ElementTree,Element
import xml.etree.ElementTree as ET


import cv2
import numpy as np

for ii in range(1000,1217):

    pth = '/Users/weng/Documents/大四/ssd_trains/JPEGImages/image'+str(ii)+'.jpg'

    xml_path = '/Users/weng/Documents/大四/ssd_trains/Annotations'
    xmlfile = xml_path + '/image'+str(ii)+'.xml'
    tree = ET.parse(xmlfile)

    root = tree.getroot()

    xmins = []

    for ele in root.iter('xmin'):
        xmins.append(int(ele.text))

    xmaxs = []
    for ele in root.iter('xmax'):
        xmaxs.append(int(ele.text))

    ymins = []
    for ele in root.iter('ymin'):
        ymins.append(int(ele.text))
    ymaxs = []
    for ele in root.iter('ymax'):
        ymaxs.append(int(ele.text))

    rects = []
    for i in range(len(xmins)):
        rects.append([xmins[i], ymins[i], xmaxs[i], ymaxs[i]])
    #
    #
    # tree.write(xml_path+'/image1的副本2.xml')

    src = cv2.imread(pth)
    img_height = 256
    img_width = 456
    scale_y = src.shape[0] / img_height
    scale_x = src.shape[1] / img_width

    if scale_x > scale_y:
        scale = scale_x
        real_w = img_width
        real_h = int(src.shape[0] / scale)
        black_img = np.zeros((img_height, img_width, 3), np.uint8)
        img = cv2.resize(src, (real_w, real_h))

        t_ = int((img_height - real_h) / 2)
        black_img[t_:t_ + real_h, :] = img
    else:
        scale = scale_y
        real_w = int(src.shape[1] / scale)
        real_h = img_height
        black_img = np.zeros((img_height, img_width, 3), np.uint8)
        img = cv2.resize(src, (real_w, real_h))
        t_ = int((img_width - real_w) / 2)
        black_img[:, t_:t_ + real_w] = img

    tmp=[]
    for abox in rects:
        if scale_x>scale_y:
            aa=[int((abox[0])/scale),int((abox[1])/scale+t_),int(abox[2]/scale),int((abox[3])/scale+t_)]
        else:
            aa=[int((abox[0])/scale+t_),int(abox[1]/scale),int((abox[2])/scale+t_),int(abox[3]/scale)]
        tmp.append(aa)

    rects=tmp
    print(rects)
    # for rect in rects:
    #     [x1, y1, x2, y2] = rect
    #     cv2.rectangle(black_img, (x1, y1), (x2, y2), 255, -1)

    cv2.imwrite('./ssd_train/JPEGImages/image'+str(ii)+'.jpg',black_img)
    for ele in root.iter('path'):
        ele.text = '../JPEGImages/image'+str(ii)+'.jpg'

    for ele in root.iter('xmin'):
        ele.text = str(int(int(ele.text)/scale+t_))
        # xmins.append(int(ele.text))

    for ele in root.iter('xmax'):
        ele.text = str(int(int(ele.text)/scale+t_))
        # xmins.append(int(ele.text))

    for ele in root.iter('ymin'):
        ele.text = str(int(int(ele.text)/scale))
        # xmins.append(int(ele.text))

    for ele in root.iter('ymax'):
        ele.text = str(int(int(ele.text)/scale))
        # xmins.append(int(ele.text))

    tree.write('./ssd_train/Annotations/image'+str(ii)+'.xml')