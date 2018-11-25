# -*- coding: utf-8 -*-


import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = '../train32/'
classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}  # 人为设定2类
# writer = tf.python_io.TFRecordWriter("train32.tfrecords")  # 要生成的文件
writer = tf.python_io.TFRecordWriter("val32.tfrecords")  # 要生成的文件



# with open('../train32/train.txt') as f:
#     for line in f.readlines():
#         pos = line.split(' ')[0]
#         index = int(line.split(' ')[1])
#         img = Image.open('../train32/'+pos)
#         img_raw = img.tobytes()  # 将图片转化为二进制格式
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         }))  # example对象对label和image数据进行封装
#         writer.write(example.SerializeToString())  # 序列化为字符串
#
#
# writer.close()
# f.close()

with open('../train32/val.txt') as f:
    for line in f.readlines():
        pos = line.split(' ')[0]
        index = int(line.split(' ')[1])
        img = Image.open('../train32/'+pos)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串


writer.close()
f.close()



# for index, name in enumerate(classes):
#     class_path = cwd + name + '/'
#     for img_name in os.listdir(class_path):
#         img_path = class_path + img_name  # 每一个图片的地址
#         try:
#             img = Image.open(img_path)
#             img_raw = img.tobytes()  # 将图片转化为二进制格式
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#             }))  # example对象对label和image数据进行封装
#             writer.write(example.SerializeToString())  # 序列化为字符串
#         except IOError:
#             print("Error: 读取文件失败",img_path)
#
# writer.close()

