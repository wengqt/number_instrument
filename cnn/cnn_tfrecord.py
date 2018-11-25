# -*- coding: utf-8 -*-
"""
@author: caokai
"""
import tensorflow as tf
import numpy as np


batch_size = 50



def read_and_decode(filename):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 32, 1])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label


# initial weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initial bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max_pool layer
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [batch_size, 64, 32, 1])
y_ = tf.placeholder(tf.float32, [batch_size, 1])

# first convolution and max_pool layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)

# second convolution and max_pool layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_4x4(h_conv2)

# 变成全连接层，用一个MLP处理
reshape = tf.reshape(h_pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数及优化算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



image, label = read_and_decode("./train32.tfrecords")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)


example = np.zeros((batch_size, 64, 32, 1))
l = np.zeros((batch_size, 1))

try:
    for i in range(20):
        for epoch in range(batch_size):
            example[epoch], l[epoch] = sess.run([image, label])  # 在会话中取出image和label
        train_step.run(feed_dict={x: example, y_: l, keep_prob: 0.5})
    print(accuracy.eval(feed_dict={x: example, y_: l, keep_prob: 0.5}))  # eval函数类似于重新run一遍，验证，同时修正

except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
coord.join(threads)


