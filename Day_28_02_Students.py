import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

def get_data_sparse(file_path, i):
    data=pd.read_csv(file_path,delimiter=';')
    enc=preprocessing.LabelEncoder()
    data['sex']=enc.fit_transform(data['sex'])
    data['school']=enc.fit_transform(data['school'])
    data['Pstatus']=enc.fit_transform(data['Pstatus'])
    data['address']=enc.fit_transform(data['address'])
    data['famsize']=enc.fit_transform(data['famsize'])
    data['guardian']=enc.fit_transform(data['guardian'])
    data['reason']=enc.fit_transform(data['reason'])
    data['Fjob']=enc.fit_transform(data['Fjob'])
    data['Mjob']=enc.fit_transform(data['Mjob'])
    data['romantic']=enc.fit_transform(data['romantic'])
    data['internet']=enc.fit_transform(data['internet'])
    data['higher']=enc.fit_transform(data['higher'])
    data['nursery']=enc.fit_transform(data['nursery'])
    data['activities']=enc.fit_transform(data['activities'])
    data['paid']=enc.fit_transform(data['paid'])
    data['famsup']=enc.fit_transform(data['famsup'])
    data['schoolsup']=enc.fit_transform(data['schoolsup'])
    x=np.float32(data.values[:,:-(4-i)])
    y=np.float32(data.values[:,-(4-i)])
    # x=preprocessing.minmax_scale(x) 틀림
    return model_selection.train_test_split(x,y)
def show_diff(preds, labels):
    preds_1=preds.reshape(-1)
    y_test_1=labels.reshape(-1)
    diff = preds_1-y_test_1
    error=np.mean(np.abs(diff))
    print("평균오차 : ", error)
def Model_Students(i):
    x_train, x_test, y_train,y_test = get_data_sparse('data/student-mat.csv', i)
    n_features = x_train.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))
    ph_x=tf.placeholder(tf.float32)
    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        sess.run(train,{ph_x:x_train})
        if i % 1000 == 0:
            print(i, sess.run(loss, {ph_x:x_train}))
    preds_test = sess.run(hx, {ph_x: x_test})
    show_diff(preds_test, y_test)
    sess.close()
Model_Students(1)
Model_Students(2)
Model_Students(3)