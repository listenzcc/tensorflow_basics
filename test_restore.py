# coding: utf-8

import tensorflow as tf
import numpy as np
import os

save_path = os.path.join('my_net', 'save_net.ckpt')

Ws = tf.Variable(np.arange(2).reshape((1, 2)),
                 dtype=tf.float32, name='weights')
bs = tf.Variable(np.arange(1), dtype=tf.float32, name='bias')
saver2 = tf.train.Saver()
sess2 = tf.Session()
saver2.restore(sess2, save_path)
print(sess2.run(Ws))
print(sess2.run(bs))
