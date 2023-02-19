import tensorflow as tf
import numpy as np

class gmm():
    def __init__(self, args):
        self._noise_shape = (args.noise_size, args.noise_size, 3)
        self._batch_size = args.batch
        self._num = 1
        self._idx = None
        self._initialize_variables()

    def _initialize_variables(self):
        cov = np.random.normal(loc=1.0, scale=1.0, size=(self._num, ) + self._noise_shape)
        mean = np.random.normal(loc=0.0, scale=1.0, size=(self._num,) + self._noise_shape)
        with tf.name_scope("train_parameter"):
            self._covs = tf.Variable(cov, dtype='float32')
            self._means = tf.Variable(mean, dtype='float32')

    def _get_variable(self):
        return self._covs, self._means


    def _get_noise(self):
        noise = self._means + tf.abs(self._covs) * \
                tf.random_normal(shape=(self._batch_size, ) + self._noise_shape)
        return noise
