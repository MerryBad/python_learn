import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('kaggle/leaf_train.csv')
test = pd.read_csv('kaggle/leaf_test.csv')

def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
train.head(1)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23)

for train_index, test_index in sss.split(train, labels):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


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
    return z, tf.nn.softmax(z)

def show_accuracy(preds, labels):
    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels)
    print('acc :', np.mean(bools == y_bools))
    return  np.mean(bools == y_bools)

def Model_titanic(x_train, x_test, y_train, y_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)

    z, hx = multi_layer(ph_x, ph_d, layers=[16,8,1], input_size=x_train.shape[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.FtrlOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 16
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

    preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})

    show_accuracy(preds_test, y_test)
    sess.close()
    return preds_test

# x_train, y_train = get_train_data()


# scaler = preprocessing.MinMaxScaler()
# scaler = preprocessing.StandardScaler()
# scaler.fit(x)
#
# x_train=scaler.transform(x)
# x_test=scaler.transform(x_test)

preds_test = Model_titanic(x_train, x_test, y_train, y_test)
# make_submission('kaggle/titanic_submission.csv',ids,preds_test)