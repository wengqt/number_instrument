import xml.etree.ElementTree as ET


import cv2
import numpy as np

ii=1112
pth = '/Users/weng/Documents/大四/NumInstrument/ssd_train/JPEGImages/image'+str(ii)+'.jpg'

xml_path = '/Users/weng/Documents/大四/NumInstrument/ssd_train/Annotations'
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

print(rects)



img = cv2.imread(pth)


for rec in rects:
    cv2.rectangle(img,(rec[0],rec[1]),(rec[2],rec[3]),(0,0,255),1)

cv2.imwrite('./test_xml.jpg',img)



from bs4 import BeautifulSoup

with open(xmlfile) as f:
    soup = BeautifulSoup(f, 'xml')
    folder = soup.folder
    objects = soup.find_all('object')  # Get a list of all objects in this image.

    # Parse the data for each object.
    for obj in objects:
        class_name = obj.find('name', recursive=False).text
        print(class_name)
        bndbox = obj.find('bndbox', recursive=False)
        xmin = int(bndbox.xmin.text)
        ymin = int(bndbox.ymin.text)
        xmax = int(bndbox.xmax.text)
        ymax = int(bndbox.ymax.text)
        print(xmin)
