import tensorflow as tf
import os
import numpy as np

HISTORY_SIZE = 4
HEIGHT = 84
WIDTH = 84
LEARNING_RATE = 0.00025
EPSILON = 0.01
MOMENTUM = 0.95

class Model:
    def __init__(self, sess, OUTPUT, NAME='main'):
        self.sess = sess
        self.height = HEIGHT
        self.width = WIDTH
        self.history_size = HISTORY_SIZE
        self.output = OUTPUT
        self.name = NAME
        self.build_network()
        self.checkpoint_file = os.path.join('checkpoints_breakput',
                                            'Breakout.ckpt-63')

    def build_network(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder('float', [None, self.height, self.width, self.history_size])
            self.Y = tf.placeholder('float', [None])
            self.a = tf.placeholder('int64', [None])

            f1 = tf.get_variable("f1", shape=[8, 8, 4, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f2 = tf.get_variable("f2", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f3 = tf.get_variable("f3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w1 = tf.get_variable("w1", shape=[7 * 7 * 64, 512], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable("w2", shape=[512, self.output], initializer=tf.contrib.layers.xavier_initializer())

            c1 = tf.nn.relu(tf.nn.conv2d(self.X, f1, strides=[1, 4, 4, 1], padding="VALID"))
            c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
            c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1, 1, 1, 1], padding='VALID'))

            l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])
            l2 = tf.nn.relu(tf.matmul(l1, w1))

            self.Q_pre = tf.matmul(l2, w2)

        a_one_hot = tf.one_hot(self.a, self.output, 1.0, 0.0)
        q_val = tf.reduce_sum(tf.multiply(self.Q_pre, a_one_hot), reduction_indices=1)

        # error를 -1~1 사이로 클립
        error = self.cliped_error(self.Y - q_val)

        self.loss = tf.reduce_mean(error)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

    def get_q(self, history):
        return self.sess.run(self.Q_pre, feed_dict={self.X: np.reshape(np.float32(history / 255.),
                                                                       [-1, 84, 84, 4])})

    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output)
        else:
            action = np.argmax(q)
        return action

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def cliped_error(self, error):
        return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)
