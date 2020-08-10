import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection
np.set_printoptions(linewidth=1000)
def get_data():
    names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    data=pd.read_csv('data/abalone.data', header=None, names=names)

    enc=preprocessing.LabelBinarizer()
    data['Sex']=enc.fit_transform(data['Sex'])

    enc=preprocessing.LabelEncoder()
    data['Rings']=enc.fit_transform(data['Rings'])
    y = np.int32(data.values[:,-1]//10)
    data.drop(['Rings'],axis=1,inplace=True)
    x = data.values

    x=preprocessing.minmax_scale(x)
    return model_selection.train_test_split(x,y,train_size=0.7)

def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # arg = np.argmax(labels)
    equals = (preds_arg == labels)

    print("acc :", np.mean(equals))

def model_Abalone(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = np.max(y_train)+1  # y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #--------------------------------------------#
    epochs = 10
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))
    for i in range(epochs):
        c=0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size
            part = indices[n1:n2]
            xx= x_train[part]
            yy= y_train[part]
            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})
        # print(i, c/n_iteration)
        np.random.shuffle(indices)

    # for i in range(1000):
    #     sess.run(train,{ph_x:x_train})
    #     if i%10 == 0:
    #         print(i, sess.run(loss,{ph_x:x_train}))

    preds_test = sess.run(hx, {ph_x:x_test})
    show_accuracy_sparse(preds_test, y_test)
    sess.close()
    return preds_test

x_train, x_test, y_train, y_test = get_data()
preds_list=[]
results = np.zeros([len(y_test), np.max(y_train)+1])
for i in range(5):
    with tf.variable_scope(str(i)):
        preds = model_Abalone(x_train, x_test, y_train, y_test)
        results += preds
        preds_list.append(np.argmax(preds,axis=1))
print('-'*30)
show_accuracy_sparse(results, y_test)

for preds_arg in preds_list:
    print(preds_arg[:30])
print('-'*30)
print(y_test[:30])