import tensorflow as tf
import numpy as np

def multiple_regression_1():
    x1 = [1, 0, 3, 0, 5]  # 공부한 시간
    x2 = [0, 2, 0, 4, 0]  # 출석한 일수
    y = [1, 2, 3, 4, 5]  # 성적

    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

       # 1         1         0
    hx = w1 * x1 + w2 * x2 + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(sess.run(loss))

    print(sess.run(hx))
    sess.close()

def multiple_regression_2():
    x = [[1, 0, 3, 0, 5],
         [0, 2, 0, 4, 0]]
    y = [1, 2, 3, 4, 5]  # 성적

    w = tf.Variable(tf.random_uniform([2]))
    b = tf.Variable(tf.random_uniform([1]))


    hx = w[0] * x[0] + w[1] * x[1] + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(sess.run(loss))
    print(sess.run(hx))
# bias를 weights에 넣기
def multiple_regression_3():
    # x = [[1, 0, 3, 0, 5],
    #      [0, 2, 0, 4, 0],
    #      [1, 1, 1, 1, 1]]
    # x = [[1, 0, 3, 0, 5],
    #     [1, 1, 1, 1, 1],
    #     [0, 2, 0, 4, 0]]
    x = [[1, 1, 1, 1, 1], # bias, dummy data
         [1, 0, 3, 0, 5],
         [0, 2, 0, 4, 0]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([3]))
    # b = tf.Variable(tf.random_uniform([1]))

    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(sess.run(loss))
    print(sess.run(w))
# hx연산을 행렬 곱셈으로 바꾸기
def multiple_regression_4():
    x = [[1., 1., 1., 1., 1.], # bias, dummy data
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1,3]))
    # b = tf.Variable(tf.random_uniform([1]))

    hx = tf.matmul(w,x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(sess.run(loss))
    sess.close()
# 3시간 공부하고 5번 출석한 학생과
# 4시간 공부하고 1번 출석한 학생의 성적 구하기
def multiple_regression_5():
    x = [[1., 1., 1., 1., 1.],  # bias, dummy data
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1., 2., 3., 4., 5.]

    w = tf.Variable(tf.random_uniform([1, 3]))
    # b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(w, ph_x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, feed_dict={ph_x: x}))
    # print(sess.run(hx, {ph_x: x}))
    print(sess.run(hx, {ph_x: [[1.,1.],
                               [3.,4.],
                               [5.,1.]]}))

    sess.close()
# 행렬곱셈에서 w와 x의 위치를 바꾸기
# 3시간 공부하고 5번 출석한 학생과
# 4시간 공부하고 1번 출석한 학생의 성적 구하기
def multiple_regression_6():
    # x = [[1., 1., 1., 1., 1.],  # bias, dummy data
    #      [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.]]
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]
    y = [[1],
         [2],
         [3],
         [4],
         [5]]

    w = tf.Variable(tf.random_uniform([3,1]))
    # b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x,w)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, feed_dict={ph_x: x}))

    print(sess.run(hx, {ph_x: [[1., 3., 5.],
                               [1., 4., 1.]]}))

    sess.close()
# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
# multiple_regression_5()
multiple_regression_6()
# import numpy as np
# a = np.arange(5)
# b = 2
# c = [2]
# print(a+b)
# print(a+c)