#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

##### 이 코드는 '텐서플로 첫걸음'(조르디 토레스)이란 책에 나온 코드를 옮긴 것입니다.

# 임의 데이터 분포 생성
num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# Linear regression 관련 변수들 설정
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# loss function
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# 모든 tf.Variable 초기화
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

# training
for step in range(100):
    sess.run(train)
    print(step, sess.run(W), sess.run(b), sess.run(loss))


plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel("x")
plt.ylabel("y")
plt.show()

