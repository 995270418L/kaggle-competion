import gzip
import struct

import numpy as np

def load_data_idx3(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = []
    for i in range(num_images):
        tmp = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        float_arr = tmp.astype(np.float32)
        images.append(np.reshape(float_arr, (image_size, 1)))
        offset += struct.calcsize(fmt_image)
    return images

def load_data_idx1(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_labels))
    offset += struct.calcsize(fmt_header)
    fmt_labels = '>B'
    labels = []
    for i in range(num_labels):
        a = struct.unpack_from(fmt_labels, bin_data, offset)
        labels.append(np.array(a))
        offset += struct.calcsize(fmt_labels)
    return labels

# 和均值归一化不同，他将特征放缩至[0,1]之间
def min_max_scale(value,min,max):
    return (value - min) / (max - min)

def load_data_wrapper():
    # 60000 * 784
    train_images = load_data_idx3('data/train-images.idx3-ubyte')
    train_images = np.multiply(train_images, 0.99609375/255.0)
    test_images = load_data_idx3('data/t10k-images.idx3-ubyte')
    max = np.max(test_images)
    test_images = [min_max_scale(t,0,max) for t in test_images]
    train_labels = load_data_idx1('data/train-labels.idx1-ubyte')
    test_labels = load_data_idx1('data/t10k-labels.idx1-ubyte')
    train_labels = [ vectorized_result(l) for l in train_labels ]
    train_data = zip(train_images, train_labels)
    test_data = zip(test_images, test_labels)
    return (train_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e