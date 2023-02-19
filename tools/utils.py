import numpy as np
import PIL.Image
import os
import tensorflow as tf
import pandas as pd
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import shutil
import imageio
import pickle
import math
import cv2

from sklearn import manifold
import matplotlib.pyplot as plt

kernel_size = 11
radius = 3

def resize(args, input_tensor, batch_size, size):
    prob = args.padding_prob
    PAD_VALUE = 128.0
    new_size = size + 32
    rnd = tf.random_uniform((), size, new_size, dtype='int32')
    rescaled = tf.image.resize_bilinear(input_tensor, [rnd, rnd])
    h_rem = new_size - rnd
    w_rem = new_size - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype='int32')
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype='int32')
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((batch_size, new_size, new_size, 3))
    padded = tf.image.resize_images(padded, [size, size])
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: padded, lambda: input_tensor)

def average_grads(grads_tower):
    average_grads = []
    for grad_and_vars in zip(*grads_tower):
        grads = []
        for g, _ in grad_and_vars:
            grads.append(g)
        grad = tf.reduce_mean(grads, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)
    return average_grads

def load_image(path, size):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((size, size))
    img = np.asarray(image).astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:, :, :3]
    return img

def get_image_com(index, imagenet_path, size):
    data_path = os.path.join(imagenet_path, 'images1000_val/attack')
    image_paths = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    image_names = [i for i in os.listdir(data_path)]
    image_paths = sorted(image_paths)
    image_names = sorted(image_names)
    labels_path = os.path.join(imagenet_path, 'dev.csv')
    data = pd.read_csv(labels_path, sep=',', skiprows=[0], names=['name', 'label', 'target'],
                       error_bad_lines=False)   # , quoting=csv.QUOTE_NONE)
    name = data['name'].values.tolist()
    label = data['label'].values.tolist()
    target = data['target'].values.tolist()

    def get(img_index):
        path = image_paths[img_index]
        print("---------path:", path)
        # x = load_image(path, size)
        x = imresize(imread(path, mode='RGB'), [size, size]).astype(np.float)
        id = name.index(image_names[img_index])
        y = label[id] - 1
        tar = target[id] - 1
        return x, y, tar, image_names[img_index]

    return get(index)

def get_image_val(index, imagenet_path, size):
    data_path = os.path.join(imagenet_path, 'val')
    image_paths = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    image_names = [i for i in os.listdir(data_path)]
    image_paths = sorted(image_paths)
    image_names = sorted(image_names)
    labels_path = os.path.join(imagenet_path, 'val.txt')

    def get(img_index):
        path = image_paths[img_index]
        print("---------path:", path)
        x = imresize(imread(path, mode='RGB'), [size, size]).astype(np.float)
        return x, image_names[img_index]

    return get(index)

def get_nips(index, imagenet_path, size):
    data_path = os.path.join(imagenet_path, 'images')
    image_paths = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    image_names = [i for i in os.listdir(data_path)]
    image_paths = sorted(image_paths)
    image_names = sorted(image_names)

    def get(img_index):
        path = image_paths[img_index]
        print("---------path:", path)
        x = imresize(imread(path, mode='RGB'), [size, size]).astype(np.float)
        return x, image_names[img_index]

    return get(index)

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

def gkern(kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


input_noise = tf.placeholder(dtype='float32', shape=(None, None, None, 3))
kernel = gkern(kernel_size, radius).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)
t_noise = tf.nn.depthwise_conv2d(input_noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
t_noise = t_noise / tf.reduce_mean(tf.abs(t_noise), [1, 2, 3], keep_dims=True)

m_noise = input_noise / tf.reduce_mean(tf.abs(input_noise), [1, 2, 3], keep_dims=True)

x_t = tf.placeholder(dtype='float32', shape=(None, None, None, 3))
arctan = tf.math.atan(x_t) * 2 / math.pi

def atan(x, sess):
    result = sess.run(arctan, feed_dict={x_t: x})
    return result

def save_parameter_adv(args, sess, u1, C1, x_init, y, adv_probs, png_name, path):
    adv_probs_arr = np.reshape(np.array(adv_probs),(-1))
    index = np.where(adv_probs_arr != y)[0]

    out_path = os.path.join(args.parameter_dir, path, png_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    name = 'C.npy'
    np.save(os.path.join(out_path, name), C1)

    name = 'u.npy'
    np.save(os.path.join(out_path, name), u1)

    size = args.image_size
    noise_shape = C1.shape
    batch_size = args.attack_batch
    batch_per_gmm = int(math.ceil(batch_size / noise_shape[0]))
    idx = []
    for i in range(batch_size):
        ix = int(i // batch_per_gmm)
        idx.append(ix)
    idx = np.reshape(idx, newshape=(batch_size,))
    onehot = to_categorical(idx, noise_shape[0])
    c_h = np.reshape(np.matmul(onehot, np.reshape(C1, (noise_shape[0], -1))), (batch_size,) + noise_shape[1:])
    u_h = np.reshape(np.matmul(onehot, np.reshape(u1, (noise_shape[0], -1))), (batch_size,) + noise_shape[1:])

    alpha = args.epsilon / args.attack_iter_num
    adv = x_init
    x_max = np.clip(x_init + args.epsilon, x_init, 255.0)
    x_min = np.clip(x_init - args.epsilon, 0.0, x_init)
    for i in range(int(args.attack_iter_num)):
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

        grad = noise_translation
        adv = adv + alpha * np.sign(grad)
        adv = np.clip(adv, x_min, x_max)
    adv = adv[0]
    adv_path = os.path.join(out_path, png_name)
    imsave(adv_path, adv.astype(np.uint8), format='png')

    # save all the adv image in a file
    out_path2 = os.path.join(args.adv_dir, path)
    if not os.path.exists(out_path2):
        os.makedirs(out_path2)
    adv_path2 = os.path.join(out_path2, png_name)
    adv_path2 = adv_path2.replace('JPEG','png')

    imsave(adv_path2, adv.astype(np.uint8), format='png')

    noise_path = os.path.join(out_path, 'noise.png')
    imsave(noise_path, (adv - x_init).astype(np.uint8), format='png')
    print(adv_path2)
    return

def record_information(args, dir, layers):
    with open(os.path.join(dir, 'information.txt'), 'w') as fileout:
        fileout.write('network ' + str(args.network) + '\n')
        fileout.write('epsilon ' + str(args.epsilon) + '\n')
        fileout.write('max_iter ' + str(args.max_iter) + '\n')
        fileout.write('is_sigmoid ' + str(args.is_sigmoid) + '\n')
        fileout.write('learn_rate ' + str(args.learn_rate) + '\n')
        fileout.write('layers ' + str(layers) + '\n')
        fileout.write('args ' + str(args) + '\n')


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def get_images(args, indices, imagenet_path=None):
    data_path = imagenet_path
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    def get(index):
        path = image_paths[index]
        x = imresize(imread(path, mode='RGB'), [ args.image_size,  args.image_size]).astype(np.float)
        return x

    images = np.zeros([len(indices), args.image_size,  args.image_size, 3])
    for i in range(len(indices)):
        x = get(indices[i])
        images[i] = x

    return images

def image_of_class(args, y, imagenet_path='/home/zhuangwz/dataset/ILSVRC2012/val'):
    """
    Gets an image of a prespecified class. To save computation time we use a
    presaved dictionary of an index for each class, but an attacker can also
    randomize this, or search for the best starting images.
    """
    im_indices = pickle.load(open("/home/zhuangwz/code/D6.25/tools/imagenet_dict.pickle", "rb"))
    # 1001 -> 1000
    images = get_images(args, im_indices[y - 1], imagenet_path)
    return images

