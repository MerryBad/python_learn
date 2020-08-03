import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
# 붓꽃 데이터 파일을 읽어오기
def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # arg = np.argmax(labels)
    equals = (preds_arg == labels)

    print("acc :", np.mean(equals))

def softmax_iris():
    iris = pd.read_csv('data\iris(150).csv')

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(iris.Species)

    # x, y 만들기 ( y는 인코딩되어야 한다 )
    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values
    # x=iris.values[:,:-1]
    x=np.float32(x)
    # 7대 3으로 나눠서 정확도 구하기

    data = model_selection.train_test_split(x,y,train_size=0.7)
    x_train, x_test, y_train, y_test = data

    print(x_train.shape,y_train.shape) # (105, 5) (105, 3)

    ph_x=tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (105, 3) = (105, 5) @ (5, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)  # 1 / (1 + tf.exp(-z))
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        if i%10 == 0:
            print(i, sess.run(loss,{ph_x:x_train}))

    preds_train = sess.run(hx, {ph_x:x_train})
    preds_test = sess.run(hx, {ph_x:x_test})

    show_accuracy(preds_train, y_train)
    show_accuracy(preds_test, y_test)
    sess.close()

def softmax_iris_sparse():
    iris = pd.read_csv('data\iris(150).csv')

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(iris.Species)

    # x, y 만들기 ( y는 인코딩되어야 한다 )
    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values
    # x=iris.values[:,:-1]
    x=np.float32(x)
    # 7대 3으로 나눠서 정확도 구하기

    data = model_selection.train_test_split(x,y,train_size=0.7)
    x_train, x_test, y_train, y_test = data

    print(x_train.shape,y_train.shape) # (105, 5) (105, 3)

    ph_x=tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = 3 # y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (105, 3) = (105, 5) @ (5, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)  # 1 / (1 + tf.exp(-z))
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        if i%10 == 0:
            print(i, sess.run(loss,{ph_x:x_train}))

    preds_train = sess.run(hx, {ph_x:x_train})
    preds_test = sess.run(hx, {ph_x:x_test})

    show_accuracy(preds_train, y_train)
    show_accuracy(preds_test, y_test)
    sess.close()

# softmax_iris()
softmax_iris_sparse()