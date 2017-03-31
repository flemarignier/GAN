import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/MNIST_data/", one_hot=True)

avg = np.mean(mnist.train.images, axis=0)


def remove_avg(images, avg=None):
    if avg is None:
        normalized = images - np.mean(images, axis=0)
    else:
        normalized = images - avg
    return normalized


def plot_img(img_1d):
    plt.imshow(img_1d.reshape(28, 28))
    plt.show()
    # plt.imshow((mnist.validation.images - mean)[0].reshape(28, 28))


INPUT_SIZE = 28
OUTPUT_SIZE = 28
INPUT_SIZE_S = INPUT_SIZE**2
OUTPUT_SIZE_S = OUTPUT_SIZE**2
MIDDLE_SIZE = 10
# original sizes
# INPUT_SIZE = 784
# MIDDLE_SIZE = 625

# Variables
x = tf.placeholder("float", [None, INPUT_SIZE_S])
y_ = tf.placeholder("float", [None, 10])

w_enc = tf.Variable(tf.random_normal([INPUT_SIZE_S, MIDDLE_SIZE], mean=0.0, stddev=0.05))
w_dec = tf.Variable(tf.random_normal([MIDDLE_SIZE, OUTPUT_SIZE_S], mean=0.0, stddev=0.05))
# w_dec = tf.transpose(w_enc) # if you use tied weights
b_enc = tf.Variable(tf.zeros([MIDDLE_SIZE]))
b_dec = tf.Variable(tf.zeros([OUTPUT_SIZE_S]))

res_mat = []
for i in range(INPUT_SIZE):  # lines
    l = [0] * OUTPUT_SIZE
    if i % 2 == 0:
        l[i // 2] = 1
    res_mat.append(l)
print(res_mat)
resize_mat = tf.constant(res_mat, dtype=np.float32)


def model(X, w_e, b_e, w_d, b_d):
    # Create the model
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)
    resize_x = X
    '''
    resize_x = tf.reshape(
        tf.matmul(tf.reshape(X, (INPUT_SIZE, INPUT_SIZE)), resize_mat),
        [-1]
    )
    '''
    return encoded, decoded, resize_x


encoded, decoded, resize_x = model(x, w_enc, b_enc, w_dec, b_dec)

# Cost Function basic term
cross_entropy = -1. * resize_x * tf.log(decoded) - (1. - resize_x) * tf.log(1. - decoded)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)


def create_res_img(epoch):
    # print utility
    # generate decoded image with test data
    test_fd = {x: remove_avg(mnist.test.images, avg), y_: mnist.test.labels}
    train_fd = {x: remove_avg(mnist.train.images, avg), y_: mnist.train.labels}
    decoded_imgs = decoded.eval(test_fd)
    decoded_imgs_train = decoded.eval(train_fd)
    print('loss (test) = ', loss.eval(test_fd))
    print('loss (train) = ', loss.eval(train_fd))

    x_test = remove_avg(mnist.test.images, avg)
    x_train = remove_avg(mnist.train.images, avg)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x_train[i].reshape(INPUT_SIZE, INPUT_SIZE))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display train
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(decoded_imgs_train[i].reshape(OUTPUT_SIZE, INPUT_SIZE))  # output
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display original
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(x_test[i].reshape(INPUT_SIZE, INPUT_SIZE))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(decoded_imgs[i].reshape(OUTPUT_SIZE, INPUT_SIZE))  # output
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('mnist_ae_normalized_{}_{}_{}.png'.format(INPUT_SIZE, MIDDLE_SIZE, epoch))


# Train
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        batch_xs = remove_avg(batch_xs, avg)
        train_step.run({x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
            train_loss = loss.eval({x: batch_xs, y_: batch_ys})
            print('  step, loss = %6d: %6.3f' % (i, train_loss))

        if i % 1000 == 0:
            create_res_img(i)

    create_res_img(i)
