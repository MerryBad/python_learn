import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection, impute


def get_data():
    names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'grade']
    data=pd.read_csv('data/car.data', header=None, names=names)

    enc = preprocessing.LabelEncoder()

    data['buying']=enc.fit_transform(data['buying'])
    data['maint']=enc.fit_transform(data['maint'])
    data['doors']=enc.fit_transform(data['doors'])
    data['persons']=enc.fit_transform(data['persons'])
    data['lug_boot']=enc.fit_transform(data['lug_boot'])
    data['safety']=enc.fit_transform(data['safety'])
    data['grade']=enc.fit_transform(data['grade'])

    x = data.values[:, :-1]
    y = data.values[:,-1]   # softmax에서는 반드시 1차원이여야 함
    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)

def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # arg = np.argmax(labels)
    equals = (preds_arg == labels)

    print("acc :", np.mean(equals))
def model_car_evaluation_sparse():
    x_train, x_test, y_train, y_test = get_data()

    n_features = x_train.shape[1]
    n_classes = np.max(y_train)+1 # y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))
    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.03)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        if i%10 == 0:
            print(i, sess.run(loss,{ph_x:x_train}))

    preds_test = sess.run(hx, {ph_x:x_test})

    show_accuracy(preds_test, y_test)
    sess.close()
model_car_evaluation_sparse()