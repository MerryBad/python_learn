import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    # print(sex.shape)                # (891,)

    sex = np.eye(2, dtype=np.float32)[sex]
    # print(sex.shape)                # (891, 2)
    # print(sex[:3])

    titanic.drop(['Sex', 'Age', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)
    titanic.info()

    x = np.hstack([titanic.values[:, 1:], sex])
    y = titanic.values[:, :1]

    return x, np.float32(y)

def get_data_test():
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    ids = titanic.PassengerId.values

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    sex = np.eye(2, dtype=np.float32)[sex]

    titanic.drop(['Sex', 'Age', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = np.hstack([titanic.values, sex])

    return x, ids

def make_submission(file_path, ids, preds):
    f=open(file_path, 'w', encoding='utf-8')
    print('PassengerId,Survived', file = f)
    for i in range(len(ids)):
        result=int(preds[i]>0.5)
        print('{},{}'.format(ids[i], result), file=f)
    f.close()

def multi_layer(ph_x, ph_d, layers, input_size):
    d = ph_x
    n_features=input_size
    for n_classes in layers:
        name = str(np.random.rand())
        w=tf.get_variable(name, shape=[n_features, n_classes],
                             initializer=tf.glorot_uniform_initializer)
        b=tf.Variable(tf.zeros([n_classes]))
        z=tf.matmul(d, w) + b
        if n_classes == layers[-1]:
            break
        r=tf.nn.relu(z)
        d=tf.nn.dropout(r, keep_prob=ph_d)
        n_features = n_classes
    return z, tf.sigmoid(z)

def show_accuracy(preds, labels):
    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels)
    print('acc :', np.mean(bools == y_bools))
    return  np.mean(bools == y_bools)

def Model_titanic(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)

    z, hx = multi_layer(ph_x, ph_d, layers=[4,2,1], input_size=x_train.shape[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 1000
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        print(i, c / n_iteration)

        preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
        preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})

    sess.close()
    return preds_valid.reshape(-1), preds_test.reshape(-1)

# x_train, y_train = get_train_data()
x, y = get_data_train()
x_test, ids = get_data_test()

# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x_train=scaler.transform(x)
x_test=scaler.transform(x_test)

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x,y, train_size=0.7)
preds_valid, preds_test = Model_titanic(x_train, y_train, x_valid, x_test)
make_submission('kaggle/titanic_submission.csv',ids,preds_test)