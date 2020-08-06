import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection, datasets

def get_data():
    # house=datasets.fetch_california_housing() # 20640, 8
    # print(house.DESCR)
    # dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
    # x=house.data
    # y=np.int32(house.target)
    x, y = datasets.fetch_california_housing(return_X_y=True)
    y=y.reshape(-1,1)

    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)

def show_diff(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    print('diff :', np.mean(abs(preds_arg-labels)))

def hosuing(x_train, x_test, y_train, y_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    # w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    # b = tf.Variable(tf.random_uniform([n_classes]))
    w = tf.get_variable('w', shape=[n_features, n_classes],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([n_classes]))
    # (124, 3) = (124, 13) @ (13, 3)
    hx = tf.matmul(ph_x, w) + b
    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 100
    n_iteration = len(x_train)//batch_size

    indices = np.arange(len(x_train))
    for i in range(epochs):
        c=0
        for j in range(n_iteration):
            n1=j*batch_size
            n2=n1+batch_size
            # print(np.array(range(n1,n2)))
            # print(indices[n1:n2])
            # print()
            # xx=x_train[n1:n2]
            # yy=y_train[n1:n2]
            part=indices[n1:n2]
            xx=x_train[part]
            yy=y_train[part]
            sess.run(train,{ph_x:xx, ph_y:yy})
            c += sess.run(loss, {ph_x:xx, ph_y:yy})
        # print(i, c/n_iteration)
        np.random.shuffle(indices)

    preds_test = sess.run(hx, {ph_x:x_test})
    show_diff(preds_test, y_test)
    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = hosuing(x_train, x_test, y_train, y_test)
        results += preds
print('-'*30)
show_diff(results/7,y_test)