import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing

# 7:3으로 75%의 정확도를 구현

def get_data():
    # 1. 데이터 불러오기
    indians = pd.read_csv('data/pima-indians.csv')
    # 2. X Y 데이터 분류하기
    x, y = np.float32(indians.values[:, :-1]), np.float32(indians.values[:, -1:])
    x= preprocessing.scale(x)
    # print(x.shape,y.shape) # (768, 8) (768,1)
    return model_selection.train_test_split(x, y, train_size=0.7)

def show_accuracy(preds, labels):
    preds = preds.reshape(-1)
    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))
    print('acc :', np.mean(bools == y_bools))
def pima_indians(x_train, x_test, y_train, y_test):

    # 3. 학습 준비
    w = tf.get_variable('w'+str(np.random.rand(1)[0]), shape=[x_train.shape[1], 1], initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)
    # (537, 1) = (537, 8) @ (8,1)
    z = tf.matmul(ph_x,w)+b
    hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.06)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 4. 학습 시작
    for i in range(1000):
        sess.run(train,{ph_x:x_train})
        # if i%100 == 0:
        #     print(i, sess.run(loss, {ph_x:x_train}))

    # 5. 예측하기
    preds = sess.run(hx,{ph_x:x_test})
    show_accuracy(preds,y_test)
    sess.close()

    return preds
x_train, x_test, y_train, y_test = get_data()
results = np.zeros(y_test.shape)
for i in range(7):
    preds = pima_indians(x_train, x_test, y_train, y_test)
    results += preds
print('-'*30)
show_accuracy(results/7,y_test)