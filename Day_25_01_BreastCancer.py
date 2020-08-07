import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

def get_data():
    bc = pd.read_csv('data/wdbc.data',
                     header=None)
    print(bc)
    bc.info()

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(bc[1])
    y = y.reshape(-1, 1)
    y = np.float32(y)           # int -> float

    bc.drop([0, 1], axis=1, inplace=True)
    bc.info()

    x = bc.values
    print(x.shape, y.shape)     # (569, 30) (569, 1)
    return model_selection.train_test_split(x,y,train_size=0.7)
def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    print('acc :', np.mean(bools == y_bools))

def BreastCancer(x_train, x_test, y_train, y_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.get_variable('w', shape=[n_features, n_classes],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([n_classes]))
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train, ph_y: y_train})
        # if i % 100 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

    preds_test = sess.run(hx, {ph_x:x_test})
    show_accuracy(preds_test, y_test)
    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = BreastCancer(x_train, x_test, y_train, y_test)
        results += preds
print('-'*30)
show_accuracy(results/7, y_test)