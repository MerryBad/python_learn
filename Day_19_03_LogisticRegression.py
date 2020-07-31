import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt
#
# 결과가 2개만 있을때 ( 0, 1 같이) 사용

#x 시그몬드 지나온 값
#y cost <- 값과 가까워질수록 0에 수렴


def logistic_regression():
    def show_sigmoid():
        def sigmoid(z):
            return 1/(1+np.e**-z)
        print(np.e)
        print('-'*30)
        print(sigmoid(-1))
        print(sigmoid(0))
        print(sigmoid(1))
        print('-'*30)
        #  공부 출석
    x=[[1., 1., 2.], # 탈락
       [1., 2., 1.],
       [1., 4., 5.], # 통과
       [1., 5., 4.],
       [1., 8., 9.],
       [1., 9., 8.]]
    # y=[[1., 0.],
    #    [1., 0.],
    #    [1., 1.],
    #    [1., 1.],
    #    [1., 1.],
    #    [1., 1.]]
    y=[[0],[0],[1],[1],[1],[1]]
    # y=np.int32(y)
    y=np.float32(y)
    w = tf.Variable(tf.random_uniform([3,1]))
    # (6, 1) = (6, 3) @ (3, 1)
    z = tf.matmul(x,w)
    hx= tf.sigmoid(z)   # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    preds = preds.reshape(-1)
    print(preds)

    bools=(preds>0.5)
    print(bools)

    y_bools=y.reshape(-1)
    print(y_bools)

    print('acc :', np.mean(bools == y_bools))
    sess.close()

    def show_logistic():
        def log_a():
            return 'A'
        def log_b():
            return 'B'
        y=0
        print(y*log_a()+(1-y)*log_b())
        y=1
        print(y*log_a()+(1-y)*log_b())
    show_sigmoid()
    # show_logistic()
# iris 데이터로부터 2개의 품종을 골라서
# 70%학습 30%테스트
def logistic_regression_iris():
    x, y=datasets.load_iris(return_X_y=True)
    y=y.reshape(-1,1)
    # print(x.shape, y.shape) # (150, 4) (150,)
    x=np.float32(x[:100])
    y=np.float32(y[:100])
    # print(x.shape, y.shape) #(100, 4) (100,)

    ph_x = tf.placeholder(tf.float32)

    data=model_selection.train_test_split(x,y,train_size=0.7)
    x_train, x_test, y_train, y_test = data

    w = tf.Variable(tf.random_uniform([4, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    # (100, 1) = (100, 4) @ (4, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train,{ph_x:x_train})
        print(i, sess.run(loss, {ph_x:x_train}))

    preds = sess.run(hx,{ph_x:x_test})
    preds = preds.reshape(-1)
    print(preds)

    bools = np.int32(preds > 0.5)
    print(bools)

    y_bools = np.int32(y_test.reshape(-1))
    print(y_bools)

    print('acc :', np.mean(bools == y_bools))
    sess.close()

    plt.scatter(x[:,0],x[:,2], c=y.reshape(-1))
    plt.show()


# logistic_regression()
logistic_regression_iris()