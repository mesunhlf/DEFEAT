from nets.inception_v3_imagenet import model as inception_v3
from nets.inception_resnet_v2_imagenet import model as inception_resnet_v2
from nets.resnet_v2_152_imagenet import model as resnet_v2_152
from nets.densenet_imagenet import model as densenet161
from tools.utils import *


def gkern(kernlen=21, nsig=3):
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def load_models_network(args, sess, adv_image):
    net_list = args.network

    logits = {}
    endpoints = {}
    probs = {}
    for net in net_list:
        if net == 'InceptionV3':
            lg, ep, pb = inception_v3(sess, adv_image)
        elif net == 'InceptionV4':
            lg, ep, pb = inception_v4(sess, adv_image)
        elif net == 'InceptionResnetV2':
            lg, ep, pb = inception_resnet_v2(sess, adv_image)
        elif net == 'resnet_v2_152':
            lg, ep, pb = resnet_v2_152(sess, adv_image)
        elif net == 'densenet161':
            lg, ep, pb = densenet161(sess, adv_image)
        logits[net] = lg
        endpoints[net] = ep
        probs[net] = pb

    return logits, endpoints, probs


def cos_value(x1,x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1)))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2)))
    x1_x2 = tf.reduce_sum(tf.multiply(x1, x2))
    cosin = x1_x2 / (x1_norm * x2_norm)
    return cosin

# loss function
# the cosine distance between clean and adv
def layer_cosine(args, sess, adv_image, labels, layers, endpoint_pre=None):
    net_list = args.network
    is_sigmoid = args.is_sigmoid
    logits, endpoints, probs = load_models_network(args, sess, adv_image)

    length = adv_image.shape[0]
    loss_list = []
    loss = 0
    cur_i = 0

    if(endpoint_pre is None):
        endpoint_pre = endpoints

    for net_name in net_list:
        if 'Inception' in net_name or 'efficient' in net_name:
            pre = ''
        else:
            pre = (net_name + '/')
        #pre = (net_name + '_1/') if 'vgg' in net_name else pre

        # pool layer
        temp = [0 for _ in range(length)]
        for i in range(len(layers[net_name])):
            layer = pre + layers[net_name][i]
            pool = endpoints[net_name][layer]
            if endpoint_pre is not None:
                pool_pre = endpoint_pre[net_name][pre + layers[net_name][i]]
            if is_sigmoid:
                pool = tf.nn.tanh(pool)
                pool_pre = tf.nn.tanh(pool_pre)

            for j in range(length):
                cs = cos_value(pool_pre, pool[j]) + 1e-8
                temp[j] += cs
            cur_i += 1

        loss_pool = tf.reshape(temp, shape=(length, )) / (len(net_list))
        loss = loss + loss_pool
        loss_list.append(loss_pool)


    loss /= len(net_list)
    loss_list = tf.reshape(loss_list, shape=(len(net_list), length))
    loss_list = tf.reduce_mean(loss_list, axis=1)
    l_t = logits[net_list[0]]
    argmax = tf.argmax(l_t, axis=len(l_t.shape)-1)
    return loss, loss_list, tf.squeeze(argmax), endpoints, logits


def xcent(args, sess, adv_image, labels, layers, endpoint_pre=None):
    net_list = args.network
    logits, endpoints, probs = load_models_network(args, sess, adv_image)

    loss = 0
    loss_list = []
    for net_name in net_list:
        loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[net_name], labels=labels[:adv_image.shape[0]])
        loss_ = loss_t

        if loss == 0:
            loss = loss_
        else:
            loss = loss + loss_
        loss_list.append(tf.reduce_mean(loss_t))
        loss_list.append(tf.reduce_mean(loss_))
    loss /= len(net_list)

    l_t = logits[net_list[0]]
    argmax = tf.argmax(l_t, axis=len(l_t.shape) - 1)

    return loss, loss_list, tf.squeeze(argmax), endpoints, logits