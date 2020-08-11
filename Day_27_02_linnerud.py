import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing, datasets

def basic():
    rud = datasets.load_linnerud()
    #dict_keys(['data', 'feature_names', 'target', 'target_names', 'frame', 'DESCR', 'data_filename', 'target_filename'])
    print(rud['feature_names']) # ['Chins', 'Situps', 'Jumps']
    print(rud['target_names']) # ['Weight', 'Waist', 'Pulse']
    print(rud['data'].shape) # (20, 3)

def show_diff(preds, labels):
    preds_1=preds.reshape(-1)
    y_test_1=labels.reshape(-1)
    diff = preds_1-y_test_1
    error=np.mean(np.abs(diff))
    print("평균오차 : ", error)
def show_diff_all(preds, labels):
    diff = preds-labels
    error=np.mean(np.abs(diff), axis=0)
    print("평균오차 : ", error)
def model_linnerud_by_one(x,y):
    n_features = x.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    hx = tf.matmul(x, w) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(101):
        sess.run(train)
        # print(i, sess.run(loss))
    preds = sess.run(hx)
    show_diff(preds, y)
    sess.close()
def model_linnerud_by_all(x,y):
    n_features = x.shape[1]
    n_classes = 3

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    hx = tf.matmul(x, w) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0) # 3개의 열을 확인해야되므로

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(101):
        sess.run(train)
        print(i, sess.run(loss))
    preds = sess.run(hx)
    show_diff_all(preds, y)
    sess.close()
def model_linnerud_by_all_2(x,y):
    n_features = x.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    hx = tf.matmul(x, w) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0) # 3개의 열을 확인해야되므로

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(101):
        sess.run(train)
        print(i, sess.run(loss))
    preds = sess.run(hx)
    show_diff_all(preds, y)
    sess.close()

x,y=datasets.load_linnerud(return_X_y=True)
# for i in range(3):
#     model_linnerud_by_one(np.float32(x), y[:,i:i+1])

# model_linnerud_by_all(np.float32(x),y)
model_linnerud_by_all_2(np.float32(x),y)