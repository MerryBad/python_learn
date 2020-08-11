import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)
def get_data():
    names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    data=pd.read_csv('data/abalone.data', header=None, names=names)

    enc=preprocessing.LabelBinarizer()
    data['Sex']=enc.fit_transform(data['Sex'])

    enc=preprocessing.LabelEncoder()
    data['Rings']=enc.fit_transform(data['Rings'])
    y = np.int32(data.values[:,-1:])
    data.drop(['Rings'],axis=1,inplace=True)
    x = data.values

    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)

def show_diff(preds, labels):
    preds_1=preds.reshape(-1)
    y_test_1=labels.reshape(-1)
    diff = preds_1-y_test_1
    error=np.mean(np.abs(diff))
    return error
def model_abalone_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))
    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses, errors = [], []
    for i in range(1001):
        sess.run(train, {ph_x: x_train})

        if i % 10 == 0:
            c = sess.run(loss, {ph_x: x_train})
            print(i, c)
        preds_test = sess.run(hx, {ph_x: x_test})
        error = show_diff(preds_test, y_test)

        losses.append(c)
        errors.append(error)
    plt.plot(range(len(errors)), losses, label='loss')
    plt.plot(range(len(errors)), errors, label='error')
    plt.legend()
    plt.show()
    sess.close()
x_train, x_test, y_train, y_test = get_data()
model_abalone_regression(x_train, x_test, y_train, y_test)