import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

# 문제 6
# Age 컬럼(결측치)을 데이터에 추가하세요

# 문제 7
# Embarked와 Fare 컬럼(결측치)을 데이터에 추가하세요
# (최빈값 사용)

# 문제 8
# 기존의 피처를 사용해서 새로운 피처를 추가하세요

def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    # print(titanic['Age'].mean())
    # print(titanic['Age'].mode())
    # print(titanic['Age'].median())
    median = titanic['Age'].median()
    titanic.Age.fillna(median, inplace=True)
    titanic.Embarked.fillna('S', inplace=True)

    titanic.info()
    # print(titanic.Age.describe())

    # print(titanic['Embarked'].mode())
    # print(titanic['Embarked'].value_counts())
    # print(titanic['Fare'].value_counts())
    # print(titanic['Fare'].mean())
    # exit(-1)

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    # print(sex.shape)                # (891,)

    sex = np.eye(2, dtype=np.float32)[sex]
    # print(sex.shape)                # (891, 2)
    # print(sex[:3])

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)

    titanic.drop(['Sex', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)
    titanic.info()

    x = np.hstack([titanic.values[:, 1:], sex, embarked])
    y = titanic.values[:, :1]

    return x, np.float32(y), median

def get_data_test(median):
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    titanic.Age.fillna(median, inplace=True)
    # titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(32.20, inplace=True)    # mean

    ids = titanic.PassengerId.values

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    sex = np.eye(2, dtype=np.float32)[sex]

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)

    titanic.drop(['Sex', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = np.hstack([titanic.values, sex, embarked])

    return x, ids

# 새로운 컬럼 추가
def get_data_train_added():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    # print(titanic['Age'].mean())
    # print(titanic['Age'].mode())
    # print(titanic['Age'].median())
    median = titanic['Age'].median()
    titanic.Age.fillna(median, inplace=True)
    titanic.Embarked.fillna('S', inplace=True)

    titanic.loc[titanic['Age'] <= 15, 'Age'] = 0
    titanic.loc[(titanic['Age'] > 15) & (titanic['Age'] <= 20), 'Age'] = 1
    titanic.loc[(titanic['Age'] > 20) & (titanic['Age'] <= 26), 'Age'] = 2
    titanic.loc[(titanic['Age'] > 26) & (titanic['Age'] <= 28), 'Age'] = 3
    titanic.loc[(titanic['Age'] > 28) & (titanic['Age'] <= 35), 'Age'] = 4
    titanic.loc[(titanic['Age'] > 35) & (titanic['Age'] <= 45), 'Age'] = 5
    titanic.loc[titanic['Age'] > 45, 'Age'] = 6

    titanic['Family'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])
    # titanic['AgeBin'] = pd.cut(titanic.Age, 5, labels=[0, 1, 2, 3, 4])

    titanic.info()
    # print(titanic.Age.describe())

    # print(titanic['Embarked'].mode())
    # print(titanic['Embarked'].value_counts())
    # print(titanic['Fare'].value_counts())
    # print(titanic['Fare'].mean())

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    # print(sex.shape)                # (891,)

    sex = np.eye(2, dtype=np.float32)[sex]
    # print(sex.shape)                # (891, 2)
    # print(sex[:3])

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)
    pclass = lb.fit_transform(titanic.Pclass)

    titanic.drop(['Sex', 'Cabin', 'Embarked', 'Pclass',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    titanic.info()
    print(titanic.columns.values)
    # ['Survived' 'Pclass' 'Age' 'SibSp' 'Parch' 'Fare']
    # print(titanic.Pclass.value_counts())
    # exit(-1)

    x = np.hstack([titanic.values[:, 1:], sex, embarked, pclass])
    y = titanic.values[:, :1]

    return x, np.float32(y), median


def get_data_test_added(median):
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    titanic.Age.fillna(median, inplace=True)
    # titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(32.20, inplace=True)    # mean

    titanic.loc[titanic['Age'] <= 15, 'Age'] = 0
    titanic.loc[(titanic['Age'] > 15) & (titanic['Age'] <= 20), 'Age'] = 1
    titanic.loc[(titanic['Age'] > 20) & (titanic['Age'] <= 26), 'Age'] = 2
    titanic.loc[(titanic['Age'] > 26) & (titanic['Age'] <= 28), 'Age'] = 3
    titanic.loc[(titanic['Age'] > 28) & (titanic['Age'] <= 35), 'Age'] = 4
    titanic.loc[(titanic['Age'] > 35) & (titanic['Age'] <= 45), 'Age'] = 5
    titanic.loc[titanic['Age'] > 45, 'Age'] = 6

    titanic['Family'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])

    ids = titanic.PassengerId.values

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    sex = np.eye(2, dtype=np.float32)[sex]

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)
    pclass = lb.fit_transform(titanic.Pclass)

    titanic.drop(['Sex', 'Cabin', 'Embarked', 'Pclass',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = np.hstack([titanic.values, sex, embarked, pclass])

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

    z, hx = multi_layer(ph_x, ph_d, layers=[11,5,1], input_size=x_train.shape[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 101
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
        # if i % 100 ==0:
        #     print(i, c / n_iteration)

        preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
        preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})

    sess.close()
    return preds_valid.reshape(-1), preds_test.reshape(-1)

# x_train, y_train = get_train_data()
# x, y, age= get_data_train()
# x_test, ids = get_data_test(age)
x, y, age= get_data_train_added()
x_test, ids = get_data_test_added(age)


# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x_train=scaler.transform(x)
x_test=scaler.transform(x_test)

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x,y, train_size=0.7)

preds_valid, preds_test = Model_titanic(x_train, y_train, x_valid, x_test)
# show_accuracy(preds_valid, y_valid.reshape(-1))

results_valid = np.zeros(len(x_valid))
results_test = np.zeros(len(x_test))
for i in range(5):
    with tf.variable_scope(str(i)):
        preds_valid, preds_test = Model_titanic(x_train, y_train, x_valid, x_test)
        show_accuracy(results_valid, y_valid.reshape(-1))
        results_valid += preds_valid
        results_test += preds_test
print('-'*30)
show_accuracy(results_valid/7, y_valid.reshape(-1))

make_submission('kaggle/titanic_submission.csv', ids, results_test/7)
