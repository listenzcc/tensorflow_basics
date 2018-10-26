# coding: utf-8

import tensorflow as tf
import numpy as np
import os

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='bias')
W = tf.Variable(tf.random_uniform(
    [1, 2], -1.0, 1.0), dtype=tf.float32, name='weights')
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
save_path = os.path.join('my_net', 'save_net.ckpt')
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step)
            print(sess.run(W))
            print(sess.run(b))
    save_path = saver.save(sess, save_path)
    print('Save to path: ', save_path)
