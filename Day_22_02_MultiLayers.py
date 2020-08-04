import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)

    equals = (preds_arg == labels)

    print("acc :", np.mean(equals))

# mnist = input_data.read_data_sets('mnist', one_hot=True)

# print(mnist.train.images.shape)          # (55000, 784)
# print(mnist.validation.images.shape)     # (5000, 784)
# print(mnist.test.images.shape)           # (10000, 784)

# print(mnist.train.labels.shape)          # (55000,)
# print(mnist.train.labels[:5])            # [7 3 4 6 1]

# train 데이터로 학습하고 test데이터에 대해 정확도 구하기
mnist = input_data.read_data_sets('mnist')
# print(mnist.train.images.dtype)#float32
# print(mnist.train.labels.dtype)#uint8
x_train = mnist.train.images
x_test = mnist.test.images
y_train = np.int32(mnist.train.labels)
y_test = mnist.test.labels

ph_x = tf.placeholder(tf.float32)
ph_y = tf.placeholder(tf.int32)
n_features = x_train.shape[1]
n_classes = 10  # y_train.shape[1]
w = tf.Variable(tf.random_uniform([n_features, n_classes]))
b = tf.Variable(tf.random_uniform([n_classes]))

# (55000, 10) = (55000, 784) @ (784, 10)
z = tf.matmul(ph_x, w) + b
hx = tf.nn.softmax(z)  # 1 / (1 + tf.exp(-z))
loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss=loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs=10
batch_size=100
n_iteration=len(x_train)//batch_size

for i in range(epochs):
    for j in range(n_iteration):
        n1=j*batch_size
        n2=n1+batch_size
        xx=x_train[n1:n2]
        yy=y_train[n1:n2]
        sess.run(train, {ph_x: xx, ph_y:yy})
        print(i, sess.run(loss, {ph_x: xx, ph_y:yy}))

preds_train = sess.run(hx, {ph_x: x_train})
preds_test = sess.run(hx, {ph_x: x_test})

show_accuracy_sparse(preds_train, y_train)
show_accuracy_sparse(preds_test, y_test)
sess.close()