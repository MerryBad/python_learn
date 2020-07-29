import pandas as pd
import tensorflow as tf
import numpy as np

# trees.csv 파일을 읽어서
# Girth와 Height를 이용하여 Volume 예측하기
#  10      70
#  20      80
# 1. 파일 읽어오기
# trees = pd.read_csv('data/trees.csv')
# # print(trees)
#
# # 2. 데이터 나누기
#
# girth = trees.values[:, 1:2]
# # print(girth)
# height = trees.values[:, 2:3]
# # print(height)
# volume = trees.values[:, 3:]
# # print(volume)
#
# # 3. 변수선언
# ones=np.ones(girth.size,np.float32)
# # x = [[ones[i], *girth[i], *height[i]] for i in range(girth.size)]
# x=np.zeros([31,3])
# for i in range(girth.size):
#     x[i]=ones[i], *girth[i], *height[i]
# # print(x)
# y = volume
# # x = (31, 3)
# # w = (3, 1)
#
# # y = (31, 1)
# # y = x * w
# # (31, 1) = (31,3)*(3,1)
#
# w = tf.Variable(tf.random_uniform([3,1]))
#
# ph_x = tf.placeholder(tf.float32)
#
# hx = tf.matmul(ph_x,w)
#
# loss_i = (hx - volume) ** 2
# loss = tf.reduce_mean(loss_i)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00011)
# train = optimizer.minimize(loss=loss)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # 4. 훈련하기
# for i in range(1000):
#     sess.run(train, feed_dict={ph_x: x})
#     print(i, sess.run(loss, feed_dict={ph_x: x}))
#
# # 5. 예측하기
# print(sess.run(hx, {ph_x: [[1., 10., 70.],
#                            [1., 20., 80.]]}))
#
# sess.close()

def get_trees_wx():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    girth = trees.Girth.values
    height = trees.Height.values
    volume = trees.Volume.values
    return np.float32([[1]*len(girth), girth, height]), volume

def multiple_regression_wx():
    x,y=get_trees_wx()

    w = tf.Variable(tf.random_uniform([1,3]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(w,ph_x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, feed_dict={ph_x: x}))

    print(sess.run(hx, {ph_x: [[1., 1.],
                               [10., 20.],
                               [70., 80.]]}))

    sess.close()

def get_trees_xw():
    x, y = get_trees_wx()
    return x.transpose(), y.reshape(-1,1)
    # trees = pd.read_csv('data/trees.csv', index_col=0)
    #
    # girth = trees.Girth.values.reshape(-1,1)
    # height = trees.Height.values.reshape(-1,1)
    # volume = trees.Volume.values.reshape(-1,1)
    # bias = np.ones([len(girth),1], np.float32)
    #
    # return np.hstack([bias, girth, height]), volume

def multiple_regression_xw():
    x,y=get_trees_xw()

    w = tf.Variable(tf.random_uniform([3,1]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x,x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, feed_dict={ph_x: x}))

    print(sess.run(hx, {ph_x: [[1., 10., 70.],
                               [1., 20., 80.]]}))

    sess.close()

# multiple_regression_wx()
multiple_regression_xw()