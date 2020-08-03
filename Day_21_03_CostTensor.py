import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#파이썬으로 만들었던 cost 그래프를 텐서플로 버전으로 변환하기

def cost_tensor_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)

    ph_w=tf.placeholder(tf.float32)

    hx = ph_w * x
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(-30, 50):
        w=i/10
        c = sess.run(loss, {ph_w:w})
        plt.plot(w,c, 'ro')
    plt.show()

# 반복문 없애기
def cost_tensor_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3,5,0.1).reshape(-1,1)
    hx = w * x

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    c = sess.run(loss)
    plt.plot(w, c, 'ro')
    plt.show()

def cost_tensor_dummy():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3,5,0.1).reshape(-1,1)
    hx = w * x

    loss_i = (hx - y) ** 2
    c = np.mean(loss_i, axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.plot(w, c, 'ro')
    plt.show()

# cost_tensor_1()
# cost_tensor_2()
cost_tensor_dummy()