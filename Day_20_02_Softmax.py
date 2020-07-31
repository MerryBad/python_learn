import numpy as np
import tensorflow as tf

# 합계가 1.0
#
def show_softmax():
    def softmax(a):
        b=np.e ** a
        return b/np.sum(b)
    a=np.float32([2.0, 1.0, 0.1])

    print(a/np.sum(a))
    print(softmax(a))

def softmax_regression():
    #      공부 출석
    x=[[1., 1., 2.], # C
       [1., 2., 1.],
       [1., 4., 5.], # B
       [1., 5., 4.],
       [1., 8., 9.], # A
       [1., 9., 8.]]

    y=[[0,0,1],[0,0,1],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
    y=np.float32(y)

    w = tf.Variable(tf.random_uniform([3,3]))
    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x,w)
    hx= tf.nn.softmax(z)   # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        # print(i, sess.run(loss))

    preds = sess.run(hx)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)
    y_arg = np.argmax(y, axis=1)
    print(y_arg)

    equals = (preds_arg == y_arg)
    print(equals)

    print("acc :" , np.mean(equals))
    sess.close()

# show_softmax()
softmax_regression()