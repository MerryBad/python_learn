import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection, impute


def get_data():
    # #    1. Sample code number            id number
    # #    2. Clump Thickness               1 - 10
    # #    3. Uniformity of Cell Size       1 - 10
    # #    4. Uniformity of Cell Shape      1 - 10
    # #    5. Marginal Adhesion             1 - 10
    # #    6. Single Epithelial Cell Size   1 - 10
    # #    7. Bare Nuclei                   1 - 10
    # #    8. Bland Chromatin               1 - 10
    # #    9. Normal Nucleoli               1 - 10
    # #   10. Mitoses                       1 - 10
    # #   11. Class:                        (2 for benign, 4 for malignant)
    # names = ['code', 'Clump', 'Size', 'Shape', 'Adhesion',
    #          'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli',
    #          'Mitoses', 'Class']
    # bc = pd.read_csv('data/breast-cancer-wisconsin.data',
    #                  header=None,
    #                  names=names)
    # print(bc)
    # bc.info()
    #
    # counts = bc.Nuclei.value_counts()
    # print(counts)
    #
    # most_freq = counts.index[0]
    #
    # # print(bc[6])
    # # bc.drop([6], axis=1, inplace=True)
    #
    # enc = preprocessing.LabelEncoder()
    # y = enc.fit_transform(bc['Class'])
    # y = y.reshape(-1, 1)
    # y = np.float32(y)           # int -> float
    #
    # nuclei = bc.Nuclei.values
    # print(type(nuclei), nuclei.dtype)
    # print(nuclei[:5])
    # print(set(nuclei))
    # print(np.unique(nuclei))
    #
    # # 2번
    # # equals = (nuclei == '?')        # [False True True ... False]
    # # nuclei[nuclei == '?'] = '0'
    # nuclei[nuclei == '?'] = str(most_freq)
    # nuclei = np.int64(nuclei)
    # print(np.unique(nuclei))
    #
    # # 1번
    # # temp = [i if i != '?' else '0' for i in nuclei]
    # # temp = np.int64(temp)
    # # print(np.unique(temp))
    #
    # # print(bc.Nuclei)
    # bc.drop(['code', 'Nuclei', 'Class'], axis=1, inplace=True)
    #
    # # 1번
    # # bc['Nuclei'] = temp
    # # bc.info()
    #
    # x = bc.values
    #
    # # 2번
    # x = np.hstack([x, nuclei.reshape(-1, 1)])
    # print(x.shape, y.shape)     # (699, 8) (699, 1)
    # print(x.dtype)
    #
    # return model_selection.train_test_split(x, y, train_size=0.7)
    #
    data=pd.read_csv('data/breast-cancer-wisconsin.data', header=None)

    x = data.values[:, :-1]
    imp = impute.SimpleImputer(missing_values='?',strategy='most_frequent')
    imp.fit(x)
    x=np.float32(imp.transform(x))

    y=data.values[:,-1:]
    enc=preprocessing.LabelEncoder()
    y=enc.fit_transform(y)
    y=np.float32(y.reshape(-1,1))
    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)
def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    print('acc :', np.mean(bools == y_bools))

def BreastCancer(x_train, x_test, y_train, y_test):
    ph_x = tf.placeholder(tf.float32)
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.get_variable('w', shape=[n_features, n_classes],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([n_classes]))
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train,{ph_x:x_train})
        # if i % 100 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

    preds_test = sess.run(hx, {ph_x:x_test})
    show_accuracy(preds_test, y_test)
    sess.close()
    return preds_test

x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(10):
    with tf.variable_scope(str(i)):
        preds = BreastCancer(x_train, x_test, y_train, y_test)
        results += preds
print('-'*30)
show_accuracy(results/10, y_test)