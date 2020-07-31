import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing

def pima_indians():
    # 피마 인디언 당뇨병 데이터로부터
    # 70% 데이터로 학습하고 30% 데이터로 정확도를 계산하기

    # 1. 데이터 불러오기
    indians= pd.read_csv('data/pima-indians.csv')
    # print(indians.info())

    # 2. X Y 데이터 분류하기
    x, y= np.float32(indians.values[:,:-1]), np.float32(indians.values[:,-1:])
    # print(x.shape,y.shape) # (768, 8) (768,1)
    data = model_selection.train_test_split(x,y,train_size=0.7)
    x_train, x_test, y_train, y_test = data
    # print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (537, 8) (231, 8) (537,1) (231,1)



    # 3. 학습 준비
    w = tf.Variable(tf.random_uniform([x.shape[1],1]))
    b = tf.Variable(tf.random_uniform([1]))
    ph_x = tf.placeholder(tf.float32)
    # (537, 1) = (537, 8) @ (8,1)
    z = tf.matmul(ph_x,w)+b
    hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 4. 학습 시작
    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        print(i, sess.run(loss, {ph_x:x_train}))

    # 5. 예측하기
    preds = sess.run(hx,{ph_x:x_test})
    preds = preds.reshape(-1)
    print(preds)
    #
    bools = np.int32(preds > 0.5)
    print(bools)

    y_bools = np.int32(y_test.reshape(-1))
    print(y_bools)

    print('acc :', np.mean(bools == y_bools))

    sess.close()

# 학습:검증:검사 데이터로 나누고 (6:2:2)
# 5번에 걸친 학습과 검증 평균 정확도가 65%가 넘도록 만들기
def pima_indians_validation():
    # 1. 데이터 불러오기
    indians= pd.read_csv('data/pima-indians.csv')
    # print(indians.info())

    # 2. X Y 데이터 분류하기
    x, y= np.float32(indians.values[:,:-1]), np.float32(indians.values[:,-1:])
    # print(x.shape,y.shape) # (768, 8) (768,1)
    x=preprocessing.scale(x)
    # test_size=int(len(x)*0.2)
    # data = model_selection.train_test_split(x, y, test_size=test_size)
    # x_train, x_test, y_train, y_test = data
    # data = model_selection.train_test_split(x_test, y_test, test_size=test_size)
    # x_test, x_valid, y_test, y_vaild = data

    data = model_selection.train_test_split(x,y,train_size=0.6)
    x_train, x_test, y_train, y_test = data
    result=[]
    for _ in range(1):
        data = model_selection.train_test_split(x_test, y_test, train_size=0.5)
        x_test, x_check, y_test, y_check = data
        print(x_train.shape,x_test.shape,x_check.shape,y_train.shape,y_test.shape,y_check.shape)
        #        (460, 8)     (154, 8)      (154, 8)      (460, 1)     (154, 1)     (154, 1)

        # 3. 학습 준비
        w = tf.Variable(tf.random_uniform([x.shape[1],1]))
        b = tf.Variable(tf.random_uniform([1]))
        ph_x = tf.placeholder(tf.float32)
        # (460, 1) = (460, 8) @ (8,1)
        z = tf.matmul(ph_x,w)+b
        hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

        # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
        loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
        loss = tf.reduce_mean(loss_i)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003)
        train = optimizer.minimize(loss=loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            sess.run(train,{ph_x:x_train})
            # if i % 10 == 0:
            #   print(i, sess.run(loss, {ph_x:x_train}))

        preds = sess.run(hx,{ph_x:x_test})
        preds = preds.reshape(-1)
        # print(preds)

        bools = np.int32(preds > 0.5)
        # print(bools)

        y_bools = np.int32(y_test.reshape(-1))
        # print(y_bools)

        acc=np.mean(bools == y_bools)
        print('acc :', acc)
        result.append(acc)
        sess.close()
    # print(result)
    print(np.mean(result))

# pima_indians()
pima_indians_validation()