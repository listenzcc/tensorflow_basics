# coding: utf-8

import tensorflow as tf
import os

save_path = 'example_save_path'
save_model = 'my_test_model'
model = os.path.join(save_path, save_model)

# Creat container firstly
k = tf.Variable(tf.random_uniform([1]), dtype=tf.float32, name='k')
b = tf.Variable(tf.random_uniform([1]), dtype=tf.float32, name='b')

# Init sess and saver
sess = tf.Session()
saver = tf.train.Saver()

# Restore from model
saver.restore(sess, model)

# See what we've got
print(sess.run(k), end=', ')
print(sess.run(b))
