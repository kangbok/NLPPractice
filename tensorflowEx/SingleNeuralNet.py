#-*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 예제 데이터 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.convert_to_tensor(mnist.train.images).get_shape()

x = tf.placeholder("float", [None, 784]) # x 지정
W = tf.Variable(tf.zeros([784, 10])) # 가중치 메트릭스 W
b = tf.Variable(tf.zeros([10])) # 바이어스 벡터 b
y = tf.nn.softmax(tf.matmul(x, W) + b) # softmax 결과
y_ = tf.placeholder("float", [None, 10]) # 정답 값

cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # y와 y_의 cross entropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 훈련용 데이터셋에서 무작위 100개 추출
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
