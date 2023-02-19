import time
from math import ceil
from tools.utils import *
from tools.loss import *
from tools.gmm import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

def train(args, opt_layers, path):
    img_index = args.img_index
    batch_size = args.batch
    img_num = args.img_num
    epsilon = args.epsilon
    max_iter = args.max_iter
    num_gpu = args.num_gpu
    image_size = args.image_size

    # generate guassian kernel
    def gkern(kernlen=21, nsig=3):
        import scipy.stats as st
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel_size = args.kernel_setup[0]
    radius = args.kernel_setup[1]
    kernel = gkern(kernel_size, radius).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)

    # placeholder for input images and labels
    dis = ceil(batch_size / num_gpu)
    x = tf.placeholder(shape=[image_size, image_size, 3], dtype='float32', name='input_image')
    x_other = tf.placeholder(shape=[image_size, image_size, 3], dtype='float32', name='x_other')
    labels = tf.placeholder(shape=[batch_size, 1000], dtype='float32', name='input_label')
    ini_image = tf.placeholder(shape=[1, image_size, image_size, 3], dtype='float32', name='initial_image')
    ini_label = tf.placeholder(shape=[1, 1000], dtype='float32', name='initial_label')

    # define the gmm models
    gmm_model = gmm(args)
    # get train paramater
    C, u = gmm_model._get_variable()

    def atan(x):
        return (tf.math.atan(x) * 2 / math.pi)


    normalize = atan
    noise = gmm_model._get_noise()

    # resize -> image_size
    if noise.shape[1] != image_size:
        noise = tf.image.resize_bilinear(noise, size=(image_size, image_size))

    noise = normalize(noise) * epsilon
    if args.translation_type != 'none':
        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
        noise = normalize(noise) * epsilon

    if args.mixup_ratio < 1.0:
        x_mix = x * args.mixup_ratio + tf.expand_dims(x_other, 0) * (1-args.mixup_ratio)
    else:
        x_mix = x

    adv_image = tf.add(x_mix, noise)

    if args.padding_prob > 0.0:
        adv_pad = resize(args, adv_image, batch_size, image_size)
    else:
        adv_pad = adv_image

    adv_clip = tf.clip_by_value(adv_pad, 0.0, 255.0)

    f_loss, f_loss_list, f_pred, f_endpoint, f_logits = xcent(args, sess, ini_image, labels=ini_label,
                                                    layers=None)

    # optimizer
    t_vars = tf.trainable_variables(scope="train_parameter")
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learn_rate)
    grads_tower = []
    losses = []
    losses_list = []
    predicts = []

    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i):
            print('loading on gpu %d of %d' % (i + 1, num_gpu))
            adv = adv_clip[i*dis: (i+1)*dis]
            label_per_gpu = labels[i*dis: (i+1)*dis]
            loss, loss_list, pred, endpoints, logits = layer_cosine(args, sess, adv, labels=label_per_gpu,
                                            layers=opt_layers, endpoint_pre=f_endpoint)
            grads_tower.append(optimizer.compute_gradients(loss, t_vars))
            losses.append(loss)
            losses_list.append(loss_list)
            predicts.append(pred)


    d_loss = tf.zeros(shape=(1,), dtype='float32')
    d_loss_list = tf.zeros(shape=(1,), dtype='float32')

    # losses & grads
    losses_est = tf.reshape(losses, shape=(1, batch_size))[0]
    losses_list_est = tf.div(tf.reduce_sum(losses_list, axis=0), args.num_gpu)
    grads = average_grads(grads_tower)

    update = optimizer.apply_gradients(grads)

    # c, u initializer
    var_init = tf.variables_initializer(t_vars)
    opt_init = tf.variables_initializer(optimizer.variables())
    sess.run(tf.local_variables_initializer())

    def updater(x_init, mix_image, label):
        sess.run([update],
                 feed_dict={x: x_init, x_other: mix_image, labels: label, ini_image: np.expand_dims(x_init, 0)})

    def show_log(x_init, mix_image, label):
        adv_np, noise_np, C_np, u_np, ls, dloss, dloss_list, ls_list, pds = sess.run(
            [adv_image, noise, C, u, losses_est, d_loss, d_loss_list, losses_list_est, predicts],
            feed_dict={x: x_init, x_other: mix_image, labels: label, ini_image: np.expand_dims(x_init, 0)})
        print("d_loss:", dloss, 'd_list:',dloss_list)
        return adv_np, noise_np, C_np, u_np, ls, ls_list, pds

    parameter_dir = os.path.join(args.parameter_dir, path)
    adv_dir = os.path.join(args.adv_dir, path)

    if not os.path.exists(parameter_dir):
        os.makedirs(parameter_dir)

    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)

    record_information(args, parameter_dir, opt_layers)

    mix_zeros = np.zeros(shape=x.shape)
    for p in range(img_num):
        print('image no:', p)
        time1 = time.time()

        sess.run(opt_init)
        sess.run(var_init)

        x_init, y, target, png_name = get_image_com(img_index, "/home/zhuangwz/dataset/ali2019/", image_size)

        gt_labels = np.zeros([batch_size, 1000], dtype='float32')
        y, l  = sess.run([f_pred, f_logits], feed_dict={x: x_init / args.mixup_ratio, x_other: mix_zeros,
                                       labels: gt_labels, ini_image: np.expand_dims(x_init, 0)})
        print('[substitude model] initial predict:', y)
        gt_labels[:, y] = 1.0
        prd = y
        img_index += 1

        mixup_images = image_of_class(args, prd)

        for j in range(max_iter):
            mix_num = len(mixup_images)
            rnd = np.random.randint(0, mix_num)
            updater(x_init, mixup_images[rnd], gt_labels)
            if (j+1) % 30 == 0 and (j+1) % 20 != 0:
                adv_np, noise_np, C_np, u_np, ls, ls_list, pds = \
                    show_log(x_init, mixup_images[rnd], gt_labels)
                not_flip = np.sum(np.array(pds) == y)
                print("step %d: %d / %d" % (j + 1, batch_size - not_flip, batch_size))
                print("adv predict: ", np.reshape(np.array(pds), (-1)))

        adv_np, noise_np, C_np, u_np, ls, ls_list, pds = show_log(x_init, mixup_images[rnd], gt_labels)

        print("adv predict: ", np.reshape(np.array(pds), (-1)))
        save_parameter_adv(args, sess, u_np, C_np, x_init, y, pds, png_name, path)
        print('total time %.4f', time.time() - time1)


