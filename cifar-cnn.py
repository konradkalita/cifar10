# Oświadczam że kod napisałem samodzielnie - Konrad Kalita

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cifar
import math

train_data, train_labels = cifar.load_training_data()
test_data, test_labels = cifar.load_test_data()
train_data = train_data[:45000]
train_labels = train_labels[:45000]
print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)


def permute(data, labels):
        perm = np.random.permutation(data.shape[0])
        return (data[perm], labels[perm])


def get_batch(data, labels, size):
    for k in range(0, data.shape[0],  size):
        yield k/size, (data[k:k + size], labels[k:k + size])


def crop(imgs, train=False):
    imgs = imgs.reshape(imgs.shape[0], 32, 32, 3)
    if train:
        for i in range(8):
            side = np.random.randint(2)
            imgs = np.delete(imgs, side*(imgs.shape[1] - 1), 1)
            side = np.random.randint(2)
            imgs = np.delete(imgs, side*(imgs.shape[2] - 1), 2)
        return imgs
    else:
        for i in range(4):
            imgs = np.delete(imgs, 0, 1)
            imgs = np.delete(imgs, imgs.shape[1] - 1, 1)
            imgs = np.delete(imgs, 0, 2)
            imgs = np.delete(imgs, imgs.shape[2] - 1, 2)
        return imgs


def conv2d(x, W, padding, name=None):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding, name=name)


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def apply_conv(x, kernel_shape, padding='SAME'):
    w = weight_variable(kernel_shape)
    x = conv2d(x, w, padding)
    x = batch_norm(x, [0, 1, 2])
    x = tf.nn.elu(x)
    return x


def res_block(x, kernel_shape):
    h = apply_conv(x, kernel_shape)
    h = apply_conv(h, kernel_shape)
    return h + x


def deepnn(x, keep_prob, training):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x = tf.cond(training, lambda: preprocess(x), lambda: x)

    # [-1, 24, 24, 3]
    x = batch_norm(x, [0, 1, 2])
    # x = tf.layers.dropout(x, 1 - keep_prob, training=training)
    # First convolutional layer - maps one grayscale image to 32 feature maps.

    x = apply_conv(x, [3, 3, 3, 32])
    x = res_block(x, [3, 3, 32, 32])
    x = res_block(x, [3, 3, 32, 32])

    # [-1, 12, 12, 32]
    # tf.Print(x, [x.shape])
    x = max_pool_2x2(x)
    x = tf.layers.dropout(x, 1 - keep_prob, training=training)

    x = apply_conv(x, [3, 3, 32, 64])
    x = res_block(x, [3, 3, 64, 64])
    x = res_block(x, [3, 3, 64, 64])
    x = res_block(x, [3, 3, 64, 64])

    # [-1, 6, 6, 64]
    # tf.Print(x, [x.shape])
    x = max_pool_2x2(x)
    x = tf.layers.dropout(x, 1 - keep_prob, training=training)

    x = apply_conv(x, [3, 3, 64, 128])
    x = res_block(x, [3, 3, 128, 128])
    x = res_block(x, [3, 3, 128, 128])

    # tf.Print(x, [x.shape])
    x = apply_conv(x, [6, 6, 128, 128], padding='VALID')

    x = tf.reshape(x, [-1, 128])
    # tf.Print(x, [x.shape])
    x = tf.layers.dropout(x, 1 - keep_prob, training=training)

    # Fully connected layer 1
    W_fc1 = weight_variable([128, 10])
    B_fc1 = bias_variable([10])

    x = tf.matmul(x, W_fc1) + B_fc1

    return x


def test_accuracy(sess, accuracy, test_batch, test_dict, x, y_):
    acc = []
    start, end = 0, test_batch
    for i in range(test_data.shape[0] // test_batch):
        test_dict[x] = crop(test_data[start:end])
        test_dict[y_] = test_labels[start:end]
        acc.append(sess.run(accuracy, feed_dict=test_dict))
        start += test_batch
        end += test_batch
    return sum(acc) / len(acc)


def batch_norm(x, axes):
    mean, var = tf.nn.moments(x, axes, keep_dims=True)
    x_norm = (x - mean) / tf.sqrt(var + 1e-4)
    tmp = x.get_shape()[3]
    tf.Print(x, [tmp])
    beta = tf.Variable(tf.constant(0.1, shape=[tmp]))
    gamma = tf.Variable(tf.constant(1., shape=[tmp]))
    return x_norm * gamma + beta


def preprocess(imgs):
    #deg_15 = math.pi / 12
    #angle = tf.random_uniform([1], -deg_15, deg_15)
    #imgs = tf.contrib.image.rotate(imgs, angle)
    imgs = tf.map_fn(tf.image.random_flip_left_right, imgs)
    return imgs


def train(learning_rate, epochs, batch_size):

    # Create the model


    x = tf.placeholder(tf.float32, [None, 24, 24, 3], name='images')
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    # Build the graph for the deep net
    y_conv = deepnn(x, keep_prob, training)
    saver = tf.train.Saver()
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = train_data.shape[0]
    steps = (num_examples - batch_size + 1) // batch_size

    test_dict = {x: crop(test_data), y_: test_labels, keep_prob: 1.0, training: False}

    # test_dict = {x: crop(test_data), y_: test_labels, keep_prob: 1.0, training: False}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            data, labels = permute(train_data, train_labels)
            with tqdm(total=steps) as pbar:
                for i, (imgs, labels) in get_batch(data, labels, batch_size):
                    train_dict = {x: crop(imgs, train=True), y_: labels, keep_prob: 0.7, training: True}
                    # print(sess.run(y_conv, feed_dict=train_dict).shape)
                    train_step.run(feed_dict=train_dict)
                    pbar.update(1)
                # train_accuracy = accuracy.eval(feed_dict={x: crop(train_data), y_: labels, keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (e, train_accuracy))
                print('epoch %d test accuracy %g' % (e, test_accuracy(sess, accuracy, 500, test_dict, x, y_)))
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)



def test():
    x = tf.placeholder(tf.float32, [None, 24, 24, 3], name='images')
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
    y_conv = deepnn(x, keep_prob, training)
    saver = tf.train.Saver()
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_dict = {x: crop(test_data), y_: test_labels, keep_prob: 1.0, training: False}
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./tmp/model.ckpt")
        # sess.run(tf.global_variables_initializer())
        print('test accuracy %g' % (test_accuracy(sess, accuracy, 500, test_dict, x, y_)))


# train(1e-3, 1, 128)
test()
