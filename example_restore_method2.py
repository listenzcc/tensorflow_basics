# coding: utf-8

import tensorflow as tf
import os

save_path = 'example_save_path'
save_model = 'my_test_model'
model = os.path.join(save_path, save_model)

# Init sess
sess = tf.Session()

# Get graph from model
saver = tf.train.import_meta_graph(model+'.meta')
saver.restore(sess, tf.train.latest_checkpoint(save_path))
graph = tf.get_default_graph()

# Restore k and b from graph
k = graph.get_tensor_by_name('k:0')
b = graph.get_tensor_by_name('b:0')

# See what we've got
print(sess.run(k), end=', ')
print(sess.run(b))
