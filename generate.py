from tools.utils import *
from PIL import Image
# from scipy.ndimage import zoom
import cv2
#from process_adv import get_labels
import copy
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.contrib.slim.python.slim.nets import resnet_v1, resnet_v2
from my_models import InceptionResnetV2
from my_models import InceptionResnetV2_ghost
from my_models import inception_v4
from my_models import inception_v3_ghost
from tensorflow.contrib.framework.python.ops import assign_from_checkpoint_fn, get_model_variables
from tensorflow.python import pywrap_tensorflow
import time
import math
import setup

def gkern(kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

input_noise = tf.placeholder(dtype='float32', shape=(None, None, None, 3))
kernel = gkern(setup.kernel_size, setup.radius).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)
t_noise = tf.nn.depthwise_conv2d(input_noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
t_noise = t_noise / tf.reduce_mean(tf.abs(t_noise), [1, 2, 3], keep_dims=True)

m_noise = input_noise / tf.reduce_mean(tf.abs(input_noise), [1, 2, 3], keep_dims=True)

def get_transfer_model(args):
    model_name = args.target_model
    scope_name = args.scope_name
    size = 224 if 'resnet_v1' in model_name else 299
    size = 224 if 'vgg' in model_name else size

    net_map = {
        'vgg_16': vgg.vgg_16,
        'vgg_19': vgg.vgg_19,
        'resnet_v1_101': resnet_v1.resnet_v1_101,
        'resnet_v1_152': resnet_v1.resnet_v1_152,
        'resnet_v2_152': resnet_v2.resnet_v2_152,
        'InceptionV3': inception_v3.inception_v3,
        'InceptionV3_ghost': inception_v3_ghost.inception_v3,
        'InceptionV4': inception_v4.inception_v4,
        'InceptionResnetV2': InceptionResnetV2.inception_resnet_v2,
        'InceptionResnetV2_ghost': InceptionResnetV2_ghost.inception_resnet_v2,
    }
    net_scope = {
        'vgg_16': vgg.vgg_arg_scope(),
        'vgg_19': vgg.vgg_arg_scope(),
        'resnet_v1_152': resnet_v1.resnet_arg_scope(),
        'resnet_v1_101': resnet_v1.resnet_arg_scope(),
        'resnet_v2_152': resnet_v2.resnet_arg_scope(),
        'InceptionV3': inception_v3.inception_v3_arg_scope(),
        'InceptionV3_ghost': inception_v3_ghost.inception_v3_arg_scope(),
        'InceptionV4': inception_v4.inception_v4_arg_scope(),
        'InceptionResnetV2': InceptionResnetV2.inception_resnet_v2_arg_scope(),
        'InceptionResnetV2_ghost': InceptionResnetV2_ghost.inception_resnet_v2_arg_scope(),
    }
    net_adv = net_map[model_name]

    adv_image = tf.placeholder(
        shape=[None, size, size, 3], dtype='float32', name='input_image')

    with slim.arg_scope(net_scope[model_name]):
        num_class = 1001 if 'resnet_v1' not in model_name else 1000
        num_class = 1000 if 'vgg' in model_name else num_class
        logits, endpoint = net_adv(adv_image, num_classes=num_class, is_training=False, scope=scope_name)
        # logits, endpoint = net_adv(adv_image, num_classes=num_class, is_training=False, scope=net_scope[model_name])
    # print(endpoint)
    logits = tf.reshape(logits, shape=(-1, num_class))

    if num_class == 1001:
        logits = logits[:, 1:]  # ignore background class
    probs = tf.math.argmax(logits, axis=len(logits.shape) - 1)

    return net_adv, adv_image, probs, logits


def transfer_initial(args, sess):
    global kernel_size
    global radius
    kernel_size = args.kernel_setup[0]
    radius = args.kernel_setup[1]
    scope_name = args.scope_name
    target_model = args.target_model
    net, ad_im, prob, logits = get_transfer_model(args)
    checkpoint_path = ''
    if ('Adv' in scope_name) or ('Ens' in scope_name):
        if 'Ghost' in scope_name:
            if target_model != 'InceptionResnetV2':
                checkpoint_path = os.path.join('my_models', 'ens3_inception_v3', 'AdvInceptionV3_ghost.ckpt')
            else:
                checkpoint_path = os.path.join('my_models', 'ens_inception_resnet_v2',
                                               'AdvInceptionResnetV2_ghost.ckpt')
        else:
            if target_model == 'InceptionV3' and 'Ens3' in scope_name:
                print("load ens3_model")
                checkpoint_path = os.path.join('my_models', 'ens3_inception_v3', 'ens3_adv_inception_v3_rename.ckpt')
            elif target_model == 'InceptionV3' and 'Ens4' in scope_name:
                print("load ens4_model")
                checkpoint_path = os.path.join('my_models', 'ens4_inception_v3', 'ens4_adv_inception_v3_rename.ckpt')
            elif target_model == 'InceptionResnetV2':
                print("load adv_incres_model")
                checkpoint_path = os.path.join('my_models', 'ens_inception_resnet_v2', 'ens_adv_inception_resnet_v2_rename.ckpt')
    else:
        checkpoint_path = os.path.join('my_models', target_model + '.ckpt')

    print(checkpoint_path)

    # var = slim.get_model_variables(scope=scope_name)
    s = tf.train.Saver(slim.get_model_variables(scope=scope_name))
    s.restore(sess, checkpoint_path)
    print('load transfer model')
    return [net, ad_im, prob, logits]

x_t = tf.placeholder(dtype='float32', shape=(None, None, None, 3))
arctan = tf.math.atan(x_t) * 2 / math.pi

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def atan(x, sess):
    result = sess.run(arctan, feed_dict={x_t: x})
    return result

def get_adv(args, path, png_name, batch_size, size, epsilon, iter_num, sess):

    init = np.load(os.path.join(path, 'init.npy'))
    # init_path = os.path.join('/home/zhuangwz/dataset/ali2019/images/',png_name)
    # init = imresize(imread(init_path, mode='RGB'), [size, size]).astype(np.float)

    y = np.load(os.path.join(path, 'y.npy'))
    C = np.load(os.path.join(path, 'C.npy'))
    u = np.load(os.path.join(path, 'u.npy'))

    noise_shape = C.shape
    if size != init.shape[0]:
        init = cv2.resize(init, (size, size))

    alpha = epsilon / iter_num
    momentum = args.momentum
    adv = np.expand_dims(copy.copy(init), 0)

    max = np.clip(init + epsilon, init, 255.0)
    min = np.clip(init - epsilon, 0.0, init)

    time1 = time.time()

    # get gmm index
    batch_per_gmm = int(math.ceil(batch_size / noise_shape[0]))
    idx = []
    for i in range(batch_size):
        ix = int(i // batch_per_gmm)
        idx.append(ix)
    idx = np.reshape(idx, newshape=(batch_size, ))
    onehot = to_categorical(idx, noise_shape[0])
    c_h = np.reshape(np.matmul(onehot, np.reshape(C, (noise_shape[0], -1))), (batch_size, ) + noise_shape[1:])
    u_h = np.reshape(np.matmul(onehot, np.reshape(u, (noise_shape[0], -1))), (batch_size, ) + noise_shape[1:])

    grad = 0
    for i in range(int(iter_num)):
        noise = u_h + c_h * np.random.normal(size=c_h.shape)
        noise = atan(noise, sess)
        if (noise.shape[1] != size):
            n_l = np.zeros((batch_size, size, size, 3), dtype='float32')
            for j in range(batch_size):
                n_l[j] = cv2.resize(noise[j], (size, size))
            noise_translation = n_l
        else:
            noise_translation = noise

        if (args.translation_type != 'none'):
            noise_translation = sess.run(t_noise, feed_dict={input_noise: noise_translation})
        else:
            noise_translation = sess.run(m_noise, feed_dict={input_noise: noise_translation})

        # noise_translation = noise
        grad = grad * momentum + noise_translation
        adv = adv + alpha * np.sign(grad)
        adv = np.clip(adv, min, max)

    print('\ntransfer time ', time.time() - time1)

    # adv = np.round(adv)
    adv = np.clip(adv, 0., 255.)
    final_noise = adv - init
    print('max:', np.max(final_noise))
    print('min:', np.min(final_noise))

    out_path = 'pic_initial'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    f = os.path.join(out_path, png_name)
    imsave(f, init.astype(np.uint8), format='png')

    out_path = 'transfer_result'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    f = os.path.join(out_path, png_name)
    imsave(f, adv[0].astype(np.uint8), format='png')

    if size == 299:
        init /= 255.0
        init = init * 2.0 - 1.0
        adv /= 255.0
        adv = adv * 2.0 - 1.0

    init = np.expand_dims(init, axis=0)
    return init, adv, y, idx
