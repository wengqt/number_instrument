
import cv2
import random
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree, objectify

outDir=''
numbers_dir=''

bg_height=192
bg_width=192

num_width=32



def gen_img():


    for i in range(-80000, -70000):
        bg = np.zeros((bg_height, bg_width), np.uint8)
        num_dots = 5
        num_1=str(-80000-i)
        num_2=str(i)


        lines_boolen= random.randint(1, 2)

        if lines_boolen ==1:
            img_name = num_1 + '.jpg'
        else:
            img_name = num_1 + '_' + num_2 + '.jpg'
        # nums = {'img_name': img_name, 'sets': [{'0':[x1,y1,x2,y2]},{}]}
        nums_log = {'img_name': img_name, 'sets': []}

        for lines in range(lines_boolen):
            if lines==1:
                nums = list(num_2)

            else:
                nums = list(num_1)


            num_img_list=[]


            for j in nums:
                if j=='-':
                    j='-1'
                num_path = '/Users/weng/Documents/大四/NumInstrument/train32/'+j+'/'
                all_list = os.listdir(num_path)

                ran = random.randint(0,len(all_list)-1)
                print(all_list[ran])
                if all_list[ran].split('.')[1]=='jpg':
                    pass
                else:
                    print(2222)
                    ran = ran-10

                nums_log['sets'].append([j])
                num_img = cv2.imread(num_path+all_list[ran],cv2.IMREAD_GRAYSCALE)
                num_img_list.append(num_img)

            diff = int((bg_width -(32*len(nums)))/2)
            if diff<0 :
                print(i)
                break
            for img_index in range(len(num_img_list)):
                # print(num_img_list)
                ymax=30+6*lines+64*(lines+1)
                ymin=30+6*lines+64*lines
                xmax=diff + 32 * (img_index+1)
                xmin=diff + 32 * img_index

                bg[30+6*lines+64*lines:30+6*lines+64*(lines+1),diff + 32 * img_index :diff + 32 * (img_index+1)] = num_img_list[img_index]
                if lines==1:
                    nums_log['sets'][img_index+len(num_1)].append([xmin, ymin, xmax, ymax])
                else:
                    nums_log['sets'][img_index].append([xmin,ymin,xmax,ymax])

        # print(nums_log)
        while num_dots:
            x1 = random.randint(0, bg_width)
            y1 = random.randint(0, bg_height)
            cv2.line(bg,(x1, y1), (x1 - 1, y1 - 1), 220, 2)
            num_dots -= 1

        cv2.imwrite('./imgs/'+img_name,bg)
        gen_xml(nums_log)

def gen_xml(logs):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('../imgs'),
        E.filename(logs['img_name']),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(str(bg_width)),
            E.height(str(bg_height)),
            E.depth('1')
        ),
        E.segmented(0),

    )
    for item in logs['sets']:
        objs=E.object(
                E.name(item[0]),
                E.pose('Unspecified'),
                E.truncated('0'),
                E.difficult('0'),
                E.bndbox(
                    E.xmin(str(item[1][0])),
                    E.ymin(str(item[1][1])),
                    E.xmax(str(item[1][2])),
                    E.ymax(str(item[1][3]))
                )
            )
        anno_tree.append(objs)

    etree.ElementTree(anno_tree).write('./xml/' + logs['img_name'] + ".xml", pretty_print=True)


if __name__ == '__main__':

    gen_img()
    # print(random.randint(1,2))