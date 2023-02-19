import os
import argparse
from defeat import train
from tools.utils import *



def main():
    parser = argparse.ArgumentParser()

    import random
    random.seed(0)

    parser.add_argument('--num_gpu', default=1,
                        help='The number of gpu to be used')

    # ************************learning************************** #
    parser.add_argument('--network', default=['InceptionV3'],
                        help='The network eg. InceptionV3, resnet_v2_152, InceptionResnetV2 , densenet161')
    parser.add_argument('--img_index', default=0,
                        help='The index of attack image')
    parser.add_argument('--img_num', default=500,
                        help='The count of imagenet image')
    parser.add_argument('--epsilon', default=16.0,
                        help='The constraint to the noise')
    parser.add_argument('--max_iter', default=30,
                        help='The max iter')
    parser.add_argument('--image_size', default=299,
                        help='The size of image')
    parser.add_argument('--noise_size', default=150,
                        help='The size of noise')
    parser.add_argument('--batch', default=3,
                        help='The batch size for guassian')
    parser.add_argument('--is_sigmoid', default=True,
                        help='The use of sigmoid for layers')
    parser.add_argument('--padding_prob', default=1.0,
                        help='use padding & resizing or not')
    parser.add_argument('--translation_type', default='guassian',
                        help='The type of loss function eg, guassian, none')
    parser.add_argument('--kernel_setup', default=[kernel_size,radius],
                        help='The kernel of guassian')
    parser.add_argument('--learn_rate', default=0.2,
                        help='The learning rate of optimizer')
    parser.add_argument('--mixup_ratio', default=0.9,
                        help='The ratio of mixup')
    parser.add_argument('--parameter_dir', default='/home/zhuangwz/code/D_result/result',
                        help='The dir for saving parameters')

    # ************************attacking************************** #
    parser.add_argument('--attack_iter_num', default=10.0,
                        help='The iter number to the noise in transfer testing')
    parser.add_argument('--attack_batch', default=1,
                        help='The batch size to use for training and testing')
    parser.add_argument('--adv_dir', default='/home/zhuangwz/code/D_result/result_adv',
                        help='The dir for saving adv images')
    args = parser.parse_args()

    method = 'DEFEAT'
    model = 'i3'

    # 'resnet_v2_152': ['block3/unit_30/bottleneck_v2']
    # 'InceptionV3': ['Mixed_7b']
    # 'InceptionResnetV2': ['Mixed_7a']
    # 'densenet161': ['dense_block4/conv_block1']

    opt_layers = {'InceptionV3': ['Mixed_7b']}
    pt = os.path.join(method, model)
    train(args, opt_layers, pt)


if __name__ == '__main__':
    main()
