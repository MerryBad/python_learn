import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

def get_data():
    wine = np.loadtxt('data/wine.data', delimiter=',')
    print(wine.shape)           # (178, 14)
    print(wine.dtype)           # float64

    x = wine[:, 1:]
    y = wine[:, 0]

    y -= 1                      # (1, 2, 3) -> (0, 1, 2)
    y = np.int32(y)             # float64 -> int32

    print(x.shape, y.shape)     # (178, 13) (178,)

    return model_selection.train_test_split(x, y, train_size=0.7)
def get_data_1():
    wine = pd.read_csv('data/wine.data', header=None)  # (178, 14)
    x = wine.values[:, 1:]  # (178, 13)
    y = wine.values[:, 0]  # (178, )
    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(y)
    # y = wine.values[:, 0]
    # y-=1
    # y=np.int32(y)
    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    print('acc :', np.mean(preds_arg == labels))

def softmax_wine(x_train, x_test, y_train, y_test):
    # print(x_train.shape, y_train.shape) # (124, 13) (124, )
    # print(x_test.shape, y_test.shape) # (54, 13) (54, )

    ph_x = tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = 3
    # w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    # b = tf.Variable(tf.random_uniform([n_classes]))
    w = tf.get_variable('w', shape=[n_features, n_classes],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([n_classes]))
    # (124, 3) = (124, 13) @ (13, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)  # 1 / (1 + tf.exp(-z))
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        # if i%10 == 0:
        #     print(i, sess.run(loss,{ph_x:x_train}))


    preds_test = sess.run(hx, {ph_x:x_test})

    show_accuracy(preds_test, y_test)
    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros([y_test.shape[0],3])
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = softmax_wine(x_train, x_test, y_train, y_test)
        results += preds
print('-'*30)
show_accuracy(results,y_test)