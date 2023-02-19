from tools.utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os
from tensorflow.contrib.framework.python.ops import assign_from_checkpoint_fn, get_model_variables
# to make this work, you need to download:
# http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# and decompress it in the `data` directory

INCEPTION_CHECKPOINT_PATH_V3 = '/home/zhuangwz/dataset/ckpt_models/inception_v3.ckpt'

def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.div(image, 255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

# input is [batch, 299, 299, 3], pixels in [0, 255]
# output is [batch, 1000]
_inception_initialized_v3 = False
def model(sess, image):
    global _inception_initialized_v3
    network_fn = _get_model(reuse=_inception_initialized_v3)
    size = network_fn.default_image_size
    image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
    image = tf.div(image, 255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    preprocessed = image
    # preprocessed = _preprocess(image, size, size)
    logits, endpoints = network_fn(preprocessed)
    logits = logits[:, 1:]  # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _inception_initialized_v3:
        print('load INCEPTION_CHECKPOINT_PATH_V3')
        # optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        load_model = assign_from_checkpoint_fn(INCEPTION_CHECKPOINT_PATH_V3, get_model_variables('InceptionV3'))
        load_model(sess)
        _inception_initialized_v3 = True

    return logits, endpoints, predictions


def _get_model_ghost(reuse):
    arg_scope = inception_v3_ghost.inception_v3_arg_scope(weight_decay=0.0)
    func = inception_v3_ghost.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

_ghost_initialized_v3 = False
def model_ghost(sess, image):
    global _ghost_initialized_v3
    network_fn = _get_model_ghost(reuse=_ghost_initialized_v3)
    size = network_fn.default_image_size
    image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
    image = tf.div(image, 255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    preprocessed = image
    # preprocessed = _preprocess(image, size, size)
    logits, endpoints = network_fn(preprocessed)
    logits = logits[:, 1:]  # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _ghost_initialized_v3:
        print('load INCEPTION_CHECKPOINT_PATH_V3')
        # optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        load_model = assign_from_checkpoint_fn(INCEPTION_CHECKPOINT_PATH_V3, get_model_variables('InceptionV3'))
        load_model(sess)
        _ghost_initialized_v3 = True

    return logits, endpoints, predictions
