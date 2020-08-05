import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))

# mnist = input_data.read_data_sets('mnist', one_hot=True)

# print(mnist.train.images.shape)          # (55000, 784)
# print(mnist.validation.images.shape)     # (5000, 784)
# print(mnist.test.images.shape)           # (10000, 784)

# print(mnist.train.labels.shape)          # (55000,)
# print(mnist.train.labels[:5])            # [7 3 4 6 1]
def baseline(ph_x):
    n_features = int(ph_x.shape[1])
    n_classes = 10  # y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (55000, 10) = (55000, 784) @ (784, 10)
    z = tf.matmul(ph_x, w) + b
    return z
def multi_layers_simple(ph_x):
    w1 = tf.Variable(tf.random_uniform([784,256]))
    b1 = tf.Variable(tf.random_uniform([256]))

    w2 = tf.Variable(tf.random_uniform([256, 256]))
    b2 = tf.Variable(tf.random_uniform([256]))

    w3 = tf.Variable(tf.random_uniform([256, 10]))
    b3 = tf.Variable(tf.random_uniform([10]))

    # (100, 256) = (100, 784) @ (784, 256)
    z1 = tf.matmul(ph_x, w1) + b1
    # (100, 256) = (100, 256) @ (256, 256)
    z2 = tf.matmul(z1, w2) + b2
    # (100, 10) = (100, 256) @ (256, 10)
    z3 = tf.matmul(z2, w3) + b3
    return z3
def multi_layers_sigmoid(ph_x):
    w1 = tf.Variable(tf.random_uniform([784,256]))
    b1 = tf.Variable(tf.random_uniform([256]))

    w2 = tf.Variable(tf.random_uniform([256, 256]))
    b2 = tf.Variable(tf.random_uniform([256]))

    w3 = tf.Variable(tf.random_uniform([256, 10]))
    b3 = tf.Variable(tf.random_uniform([10]))

    z1 = tf.matmul(ph_x, w1) + b1
    s1 = tf.sigmoid(z1)
    z2 = tf.matmul(s1, w2) + b2
    s2 = tf.sigmoid(z2)
    z3 = tf.matmul(s2, w3) + b3
    return z3
def multi_layers_relu(ph_x):
    w1 = tf.Variable(tf.random_uniform([784,256]))
    b1 = tf.Variable(tf.random_uniform([256]))

    w2 = tf.Variable(tf.random_uniform([256, 256]))
    b2 = tf.Variable(tf.random_uniform([256]))

    w3 = tf.Variable(tf.random_uniform([256, 10]))
    b3 = tf.Variable(tf.random_uniform([10]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)
    z3 = tf.matmul(r2, w3) + b3
    return z3
def multi_layers_relu_xavier(ph_x):
    w1 = tf.get_variable('w1', shape=[784, 256], initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([256]))

    w2 = tf.get_variable('w2', shape=[256, 256], initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([256]))

    w3 = tf.get_variable('w3', shape=[256, 128], initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([128]))

    w4 = tf.get_variable('w4', shape=[128, 10], initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([10]))

    w5 = tf.get_variable('w5', shape=[10, 10], initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    r3 = tf.nn.relu(z3)

    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)

    z5 = tf.matmul(r4, w5) + b5
    r5 = tf.nn.relu(z5)
    return r5

def  show_model(model, lr):
    # train 데이터로 학습하고 test데이터에 대해 정확도 구하기
    mnist = input_data.read_data_sets('mnist')
    # print(mnist.train.images.dtype)#float32
    # print(mnist.train.labels.dtype)#uint8
    x_train = mnist.train.images
    x_test = mnist.test.images
    y_train = np.int32(mnist.train.labels)
    y_test = mnist.test.labels

    ph_x = tf.placeholder(tf.float32, shape = [None, x_train.shape[1]])
    ph_y = tf.placeholder(tf.int32)
    n_features = x_train.shape[1]
    n_classes = 10  # y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))
    #--------------------#
    z = model(ph_x)
    hx = tf.nn.softmax(z)  # 1 / (1 + tf.exp(-z))
    #--------------------#

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs=10
    batch_size=100
    n_iteration=len(x_train)//batch_size

    for i in range(epochs):
        c=0
        for j in range(n_iteration):
            n1=j*batch_size
            n2=n1+batch_size
            xx=x_train[n1:n2]
            yy=y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, c / n_iteration)
    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy(preds_test, y_test)

    sess.close()

# show_model(baseline, lr=0.1)
# show_model(multi_layers_simple, lr=0.001)
# show_model(multi_layers_sigmoid, lr=0.001)
# show_model(multi_layers_relu, lr=0.01)
show_model(multi_layers_relu_xavier, lr=0.001)

