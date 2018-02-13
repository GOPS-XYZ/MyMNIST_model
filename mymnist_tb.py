#########################################################
#
# Author:   Varun Mumbarekar
# Date:     05/08/2017
# Time:     16:02
#
# MNIST model with 99.32% Test accuracy
#
#     Input Image   (28,28,1)
#    ========================
#    convl     ↓↓  (5,5,1,32)
#    ========================
#    Resultant     (28,28,32)
#    ========================
#    maxpool   ↓↓  (2 X 2)
#    ========================
#    Resultant     (14,14,32)
#    ========================
#    convl     ↓↓  (5,5,1,64)
#    ========================
#    Resultant     (14,14,64)
#    ========================
#    maxpool   ↓↓  (2 X 2)
#    ========================
#    Resultant     (7,7,64)
#    ========================
#    Softmax   ↓↓ Weight(7*7*64,1024)
#    ========================
#    Resultant     (1024 neurons)
#    ========================
#    Softmax   ↓↓ Weight(1024,10)
#    ========================
#    Resultant     (10 neurons)
#    ========================
#
# Leaning Rate: 0.0001
# Batch Size:   50
# Total Training Steps: 20000
#
#########################################################
import numpy as np
import tensorflow as tf
import timeit
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow Version: " + tf.__version__)
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)
batch_size = 50
steps = 500
learning_rate = 1e-4
#########################################################
X = tf.placeholder(tf.float32, shape=[None,28,28,1], name="Input_Image")
y_ = tf.placeholder(tf.float32, shape=[None,10], name="Output_Label")
#########################################################
def weight_variable(shape):
    weight = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name="W")
    return (weight)

def bias_varible(shape):
    bias = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name="b")
    return (bias)

def conv2d(x,W):
    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    return conv

def maxpool2x2(x):
    mpool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return mpool
#########################################################
with tf.name_scope('conv_layer1'):
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_varible([32])

    I_conv_1 = tf.nn.relu(conv2d(X, W_conv1)+b_conv1, name="conv_1")
    I_max_1 = maxpool2x2(I_conv_1)                              #Image(14*14*32)
#########################################################
with tf.name_scope('conv_layer2'):
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_varible([64])

    I_conv_2 = tf.nn.relu(conv2d(I_max_1, W_conv2)+b_conv2, name="conv_2")
    I_max_2 = maxpool2x2(I_conv_2)                              #Image(7*7*64)
#########################################################
I_max2_flat = tf.reshape(I_max_2,[-1, 7*7*64])

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_varible([1024])

    I_fc1 = tf.nn.relu(tf.matmul(I_max2_flat, W_fc1)+b_fc1)
#########################################################
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    I_fc1_drop = tf.nn.dropout(I_fc1, keep_prob)
#########################################################
with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_varible([10])

    y = tf.matmul(I_fc1_drop,W_fc2)+b_fc2
#########################################################
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.name_scope('confusion_matrix'):
#     k = tf.confusion_matrix(tf.argmax(y_, 1), tf.argmax(y, 1), num_classes=10)
########################################################
with tf.name_scope('save_model'):
    saver = tf.train.Saver()

# Start Session
with tf.Session() as sess:
    # Initial Variables
    sess.run(tf.global_variables_initializer())

    # Start Timer
    start = timeit.default_timer()

    # Write Summary of Graph for Tensorboard
    writer = tf.summary.FileWriter("graphs1/",sess.graph)
    writer.close()

    # Data for visualization
    x1 = []
    y1 = []
    y2 = []


    for i in range(steps):
        batch = mnist.train.next_batch(batch_size)

        # Training Accuracy and Cross Entropy
        if i % 20 == 0:
            a, c = sess.run([accuracy,cross_entropy], feed_dict={X:batch[0], y_:batch[1], keep_prob:1.0})
            print("Step %i => Training Accuracy: %g \tLoss: %g"%(i, a, c))

            # Data for visualization
            x1.append(i)
            y1.append(a)
            y2.append(c)

        # Training
        train_step.run(feed_dict={X:batch[0], y_:batch[1], keep_prob:0.5})

    # Test Accuracy and Cross Entropy
    # a, c, k = sess.run([accuracy, cross_entropy, k], {X: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    # print("***************************************\nStep %i => Test Accuracy: %g \tLoss: %g" %(i, a, c))

    # Confusion Matrix
    # print("k:\n", k)

    # Save Model (Weights and Biases)
    saver.save(sess,'chkpt1/MNISTmodel.ckpt')

# Time Elapsed
time = timeit.default_timer() - start
print("\nTime Elapsed: ",time)

# Display Training Accuracy and Cross Entropy
plt.subplot(2,1,1)
plt.plot(x1,y1,'r')
plt.title('Performance Graph')
plt.ylabel('Accuracy')

plt.subplot(2,1,2)
plt.plot(x1,y2,'b')
plt.xlabel('Step No.')
plt.ylabel('Cross Entropy')

plt.show()
########################################################