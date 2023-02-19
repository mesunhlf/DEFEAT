from tools.utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import densenet
import functools
import os
from tensorflow.contrib.framework.python.ops import assign_from_checkpoint_fn, get_model_variables

# to make this work, you need to download:
# http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# and decompress it in the `data` directory


INCEPTION_CHECKPOINT_161 = '/home/zhuangwz/dataset/ckpt_models/tf-densenet161.ckpt'


def _get_model(reuse):
    # print('load InceptionResnetV2')
    arg_scope = densenet.densenet_arg_scope(weight_decay=0.0)
    func = densenet.densenet161
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1000, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn


def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.div(image, 255.0)
        return image


# input is [batch, 299, 299, 3], pixels in [0, 255]
# output is [batch, 1000]
_inception_densenet = False
def model(sess, image):
    global _inception_densenet
    network_fn = _get_model(reuse=_inception_densenet)
    size = network_fn.default_image_size
    preprocessed = tf.image.resize_bilinear(image, [size, size], align_corners=False)
    preprocessed = tf.div(preprocessed, 255.0)
    logits, endpoints = network_fn(preprocessed)
    predictions = tf.argmax(logits, axis=-1)

    var = get_model_variables('densenet161')
    if not _inception_densenet:
        print('load densenet161')
        # optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH_V2)
        load_model = assign_from_checkpoint_fn(INCEPTION_CHECKPOINT_161, get_model_variables('densenet161'))
        load_model(sess)
        _inception_densenet = True

    return logits, endpoints, predictions

