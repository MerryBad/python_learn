import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

def get_data():
    names = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'native_country',
             'y'
            ]
    data=pd.read_csv('data/adult.data', header=None, names=names)

    enc=preprocessing.LabelEncoder()
    data['workclass']=enc.fit_transform(data['workclass'])
    data['education']=enc.fit_transform(data['education'])
    data['marital_status']=enc.fit_transform((data['marital_status']))
    data['occupation']=enc.fit_transform((data['occupation']))
    data['relationship']=enc.fit_transform((data['relationship']))
    data['race']=enc.fit_transform((data['race']))
    data['sex']=enc.fit_transform((data['sex']))
    data['native_country']=enc.fit_transform((data['native_country']))
    data['y']=enc.fit_transform((data['y']))
    x=data.values[:,:-1]
    y=np.float32(data.values[:,-1:])
    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)
def show_acc(preds, y):
    bools = np.int32(preds > 0.5)

    y_bools = np.int32(y.reshape(-1))

    # print('acc :', np.mean(bools == y_bools))
    return np.mean(bools == y_bools)
def Model_Adults(x_train, x_test, y_train, y_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32) # drop-out

    name = str(np.random.rand())
    w1 = tf.get_variable(name, shape=[x_train.shape[1], 64], initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([64]))

    name = str(np.random.rand())
    w2 = tf.get_variable(name, shape=[64, 32], initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([32]))

    name = str(np.random.rand())
    w3 = tf.get_variable(name, shape=[32, 32], initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([32]))

    name = str(np.random.rand())
    w4 = tf.get_variable(name, shape=[32, 16], initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([16]))

    name = str(np.random.rand())
    w5 = tf.get_variable(name, shape=[16, 1], initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_prob=ph_d)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_prob=ph_d)

    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_prob=ph_d)

    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_prob=ph_d)

    z5 = tf.matmul(d4, w5) + b5

    hx = tf.sigmoid(z5)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
    batch_size = 32
    n_iteration = len(x_train) // batch_size
    indices=np.arange(len(x_train))
    for i in range(epochs):
        c = 0
        np.random.shuffle(indices)
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size
            xx = x_train[n1:n2]
            yy = y_train[n1:n2]
            sess.run(train, {ph_x: xx, ph_y: yy, ph_d:0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d:0.7})
        # print(i, c / n_iteration)
        preds = sess.run(hx, {ph_x: x_test, ph_d:1.0})
        avg=show_acc(preds, y_test)
        print('{:3} : {:9.5f} {:9.5f}'.format(i, c/n_iteration, avg))

x_train, x_test, y_train, y_test = get_data()

Model_Adults(x_train, x_test, y_train, y_test)