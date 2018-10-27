# coding: utf-8

import tensorflow as tf
import numpy as np
import os

save_path = 'example_save_path'
save_model = 'my_test_model'

x_data = np.float32(np.random.rand(1, 100))
y_data = np.dot([0.700], x_data) + 0.300

k = tf.Variable(tf.random_uniform([1]), dtype=tf.float32, name='k')
b = tf.Variable(tf.random_uniform([1]), dtype=tf.float32, name='b')
y = tf.multiply(k, x_data) + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
saver = tf.train.Saver()

sess.run(init)
for step in range(101):
    sess.run(train)
    print(sess.run(loss), end=':\t')
    print(sess.run(k), end=', ')
    print(sess.run(b))
    if step % 20 == 0:
        saver.save(sess, os.path.join(save_path, save_model),
                   global_step=step,
                   write_meta_graph=False)

saver.save(sess, os.path.join(save_path, save_model))
