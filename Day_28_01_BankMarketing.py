import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

def get_data_sparse(file_path):
    data=pd.read_csv(file_path,delimiter=';')
    # 'age;"job";"marital";"education";"default";"balance";"housing";
    # "loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'
    # data.info()
    enc=preprocessing.LabelEncoder()
    data['job']=enc.fit_transform(data['job'])
    data['marital']=enc.fit_transform(data['marital'])
    data['education']=enc.fit_transform(data['education'])
    data['default']=enc.fit_transform(data['default'])
    data['housing']=enc.fit_transform(data['housing'])
    data['loan']=enc.fit_transform(data['loan'])
    data['contact']=enc.fit_transform(data['contact'])
    data['month']=enc.fit_transform(data['month'])
    data['poutcome']=enc.fit_transform(data['poutcome'])
    data['y']=enc.fit_transform(data['y'])
    x=np.float32(data.values[:,:-1])
    y=np.float32(data.values[:,-1:])
    # x=preprocessing.minmax_scale(x) 틀림
    return x, y
def get_data_dense(file_path):
    data=pd.read_csv(file_path,delimiter=';')
    # 'age;"job";"marital";"education";"default";"balance";"housing";
    # "loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'
    # data.info()
    enc=preprocessing.LabelBinarizer()
    data['job']=enc.fit_transform(data['job'])
    data['marital']=enc.fit_transform(data['marital'])
    data['education']=enc.fit_transform(data['education'])
    data['default']=enc.fit_transform(data['default'])
    data['housing']=enc.fit_transform(data['housing'])
    data['loan']=enc.fit_transform(data['loan'])
    data['contact']=enc.fit_transform(data['contact'])
    data['month']=enc.fit_transform(data['month'])
    data['poutcome']=enc.fit_transform(data['poutcome'])
    data['y']=enc.fit_transform(data['y'])
    x=np.float32(data.values[:,:-1])
    y=np.float32(data.values[:,-1:])
    # x=preprocessing.minmax_scale(x) 틀림
    return x, y

def show_acc(preds, y):
    bools = np.int32(preds > 0.5)

    y_bools = np.int32(y.reshape(-1))

    print('acc :', np.mean(bools == y_bools))
def Model_BankMarketing(x_train, y_train, x_test, y_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    name=str(np.random.rand())
    w1 = tf.get_variable(name, shape=[x_train.shape[1], 41], initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([41]))

    name = str(np.random.rand())
    w2 = tf.get_variable(name, shape=[41, 31], initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([31]))

    name = str(np.random.rand())
    w3 = tf.get_variable(name, shape=[31, 21], initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([21]))

    name = str(np.random.rand())
    w4 = tf.get_variable(name, shape=[21, 11], initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([11]))

    name = str(np.random.rand())
    w5 = tf.get_variable(name, shape=[11, 1], initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    r3 = tf.nn.relu(z3)

    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)

    z5 = tf.matmul(r4, w5) + b5
    r5 = tf.nn.relu(z5)

    hx = tf.sigmoid(z5)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.04)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs=30
    batch_size = 32
    n_iteration = len(x_train)//batch_size
    for i in range(epochs):
        c=0
        for j in range(n_iteration):
            n1=j*batch_size
            n2=n1+batch_size
            xx=x_train[n1:n2]
            yy=y_train[n1:n2]
            sess.run(train,{ph_x:xx, ph_y:yy})
            c+=sess.run(loss,{ph_x:xx, ph_y:yy})
        print(i, c/n_iteration)
        preds = sess.run(hx,{ph_x:x_test, ph_y:y_test})
        preds = preds.reshape(-1)
        show_acc(preds, y_test)


    sess.close()



# x_train, y_train = get_data_sparse('data/bank-full.csv')
# x_test, y_test = get_data_sparse('data/bank.csv')

x_train, y_train = get_data_dense('data/bank-full.csv')
x_test, y_test = get_data_dense('data/bank.csv')

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

Model_BankMarketing(x_train, y_train, x_test, y_test)