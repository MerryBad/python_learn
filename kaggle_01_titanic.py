import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

def get_train_data():
    file = pd.read_csv('kaggle/titanic_train.csv')
    # print(file.keys())#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                        #'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    file.drop(['PassengerId','Name', 'Age', 'Cabin', 'Embarked'],axis=1,inplace=True)
    # file.info() # Age, Cabin, Embarked 보충 필요
    enc=preprocessing.LabelEncoder()
    file['Sex']=enc.fit_transform(file['Sex'])
    file['Ticket']=enc.fit_transform(file['Ticket'])
    x=np.float32(file.values[:,1:])
    y=np.float32(file.values[:,:1])
    return x, y

def get_test_data():
    file = pd.read_csv('kaggle/titanic_test.csv')
    ids = file.PassengerId.values
    file.drop(['PassengerId', 'Name', 'Age', 'Cabin', 'Embarked'], axis=1, inplace=True)
    enc = preprocessing.LabelBinarizer()
    file['Sex'] = enc.fit_transform(file['Sex'])
    file['Ticket'] = enc.fit_transform(file['Ticket'])
    x = np.float32(file.values)

    return x, ids

def make_submission(file_path, ids, preds):
    f=open(file_path, 'w', encoding='utf-8')
    print('PassengerId,Survived', file = f)
    for i in range(len(ids)):
        result=int(preds[i]>0.5)
        print('{}, {}'.format(ids[i], result), file=f)
    f.close()
def Model_titanic(x_train, x_test, y_train):
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

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        # if i % 100 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x:x_test})
    sess.close()
    return preds.reshape(-1)

x_train, y_train = get_train_data()
x_test, ids = get_test_data()
print(x_train.shape, y_train.shape)
print(x_test.shape)

preds = Model_titanic(x_train, x_test, y_train)
make_submission('kaggle/titanic_submission.csv',ids,preds)