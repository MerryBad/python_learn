import tensorflow as tf
import numpy as np


def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    # 정답의 갯수(x)와 일치함
    hx = w * x + b
    # hx = tf.add(tf.multiply(w,x), b)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean((tf.add(tf.multiply(w,x), b) - y)**2))

    # 세션에 등록되야지 연산해줌
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        sess.run(train)
        print(sess.run(loss))

    # print(sess.run(w))
    print('-' * 20)

    # x가 5와 7일 때?

    print('5 : ', sess.run(w * 5.0 + b))
    print('7 : ', sess.run(w * 7 + b))
    print('* : ', sess.run(w * [5, 7] + b))

    ww, bb = sess.run(w), sess.run(b)
    print('5 : ', ww * 5 + bb)
    print('7 : ', ww * 7 + bb)

    # print(sess.run(w*x+b))
    # print(sess.run(w))
    # print(sess.run(b))
    # print(sess.run(hx)) # 예측값
    # print(sess.run(loss_i)) # 오차
    # print(sess.run(loss))
    # print(sess.run(train))
    sess.close()


def linear_regression_2():
    x=[1,2,3]
    y=[1,2,3]

    # w = tf.Variable(np.float32(np.random.rand(1)))
    # b = tf.Variable(np.float32(np.random.rand(1)))

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    # placeholder = 이것만 값을 바꾸고싶을때 사용
    ph_x = tf.placeholder(tf.float32)
    # ph_y = tf.placeholder(tf.float32)

    hx = w * ph_x + b

    loss_i = ( hx - y ) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, feed_dict={ph_x:x}) # placeholder에서 사용될 때 마다 알려줘야됨
        print(i, sess.run(loss, feed_dict={ph_x:x}))

    # print(sess.run(w))

    print(sess.run(hx,feed_dict={ph_x: [5,7]}))

    sess.close()

# linear_regression_1()
linear_regression_2()
