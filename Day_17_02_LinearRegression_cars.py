# 속도가 30과 50일 떄의 제동거리 구하기

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 팬더스로 파일 읽기
car = pd.read_csv('data\cars.csv')
# print(car)

# 2. x와 y 데이터 만들기
# print(car.values)

x=car.values[:,1:2] #speed
y=car.values[:,2:]  #dist

# print(x)
# print('-'*20)
# print(y)

ph_x = tf.placeholder(tf.float32)

# 3. 학습 준비하기
w = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

hx = w * ph_x + b

loss_i = ( hx - y ) ** 2
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.00376)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_print=[]
# 4. 학습하기
for i in range(1000):
    sess.run(train, feed_dict={ph_x: x})
    # print(i, sess.run(loss, feed_dict={ph_x: x}))
    loss_print.append(sess.run(loss, feed_dict={ph_x: x}))

# 5. 예측하기
print('30, 50 : ', sess.run(hx,feed_dict={ph_x: [30, 50]}))
print('7, 20 : ',sess.run(hx,feed_dict={ph_x:[7,20]}))

# 차트그리기
plt.subplot(2, 2, 1)
plt.plot(x, y, 'r')
plt.title("speed, dist")
plt.subplot(2, 2, 2)
plt.plot(range(50),sess.run(hx,{ph_x:range(50)}))
plt.title("predict ~50")
plt.subplot(2, 2, 3)
plt.title("loss ~1000")
plt.plot(range(1000), loss_print, 'b')

plt.show()
