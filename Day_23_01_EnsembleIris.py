import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
# 붓꽃 데이터 파일을 읽어오기
def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == y_arg)
    print('acc :', np.mean(equals))
def get_data():
    iris = pd.read_csv('data\iris(150).csv')

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values
    x=np.float32(x)

    return model_selection.train_test_split(x,y,train_size=0.7)

def softmax_iris(x_train, x_test, y_train, y_test):
    ph_x=tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train,{ph_x:x_train})
        # if i%10 == 0:
        #     print(i, sess.run(loss,{ph_x:x_train}))

    preds_test = sess.run(hx, {ph_x:x_test})
    show_accuracy(preds_test, y_test)
    sess.close()
    return preds_test

x_train, x_test, y_train, y_test = get_data()
results = np.zeros(y_test.shape)
# results = np.zeros_like(y_test) # int32
for i in range(7):
    preds = softmax_iris(x_train, x_test, y_train, y_test)
    results += preds
print('-'*30)
show_accuracy(results,y_test)
#앙상블 모델을 구성해서 정확도 구하기
