""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import time
import utils

#Number of filters in the first convolutional layer
no_filters1 = 16

#Number of filters in the second convolutional layer
no_filters2 = 32

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir('data')
    os.mkdir('mnist_folder')
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()
# print(label.get_shape().as_list())

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data


def convolution_layer(input, input_channels, no_filters):
    filter_shape = [5, 5, input_channels, no_filters]
    b = tf.Variable(tf.constant(0.1, shape=[no_filters]))
    filter_weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
    #strides=[1,1,1,1] will move the filter 1 pixel across x and y axis.
    #padding is set to 'SAME' to pad the input image with 0's
    conv_output = tf.nn.conv2d(input=input, filter=filter_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_output = conv_output + b
    #Applying max-pooling
    #strides=[1,2,2,1] will move the filter 2 pixel across x and y axis.
    #padding is set to 'SAME' to pad the input image with 0's
    conv_output = tf.nn.max_pool(value=conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #Applying Relu
    conv_output = tf.nn.relu(conv_output)
    # print(layer.get_shape())
    return conv_output

#Helper function to flatten the input image tensors of 4-d to 2-d
def flatten(input_image):
    no_features = input_image.get_shape()[1:4].num_elements()
    flattened_image = tf.reshape(input_image, [-1, no_features])
    return flattened_image, no_features

def fully_connected_layer(input, no_inputs, no_outputs, relu=True):
    b = tf.Variable(tf.constant(0.1, shape=[no_outputs]))
    w = tf.Variable(tf.truncated_normal([no_inputs, no_outputs], stddev=0.05))
    layer = tf.matmul(input, w) + b
    if relu:
        layer = tf.nn.relu(layer)
    return layer

#Placeholder for the Test class labels
test_label = tf.argmax(label, axis=1)

image_tensors_1 = convolution_layer(tf.reshape(img, [-1, 28, 28, 1]), 1, no_filters1)

image_tensors_2 = convolution_layer(image_tensors_1, no_filters1, no_filters2)

flattened_image, no_features = flatten(image_tensors_2)

#Relu is used here because we want to learn the non linear relationships among the image pixels
fully_conn_layer1 = fully_connected_layer(flattened_image, no_features, 128, relu=True)
#relu is not used
fully_conn_layer2 = fully_connected_layer(fully_conn_layer1, 128, 10, relu=False)

preds = tf.nn.softmax(fully_conn_layer2)

preds_classes = tf.argmax(preds, axis=1)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fully_conn_layer2, labels=label)

loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_preds = tf.equal(preds_classes, tf.argmax(label, axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/CNN', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        total_acc = 0
        try:
            while True:
                _, l,acc = sess.run([optimizer, loss,accuracy])
                total_loss += l
                n_batches += 1
                total_acc += acc
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss and training accuracy epoch {0}: {1} {2}'.format(i, total_loss / n_batches,total_acc / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    pred_class = np.zeros(shape=n_test, dtype=np.int)
    actual_class = np.zeros(shape=n_test, dtype=np.int)
    no_test_batches = 0
    sess.run(test_init)  # drawing samples from test_data
    try:
        while True:
            pred_class[no_test_batches:no_test_batches + 128], actual_class[no_test_batches:no_test_batches + 128] = sess.run([preds_classes, test_label])
            no_test_batches = no_test_batches + 128
    except tf.errors.OutOfRangeError:
        pass
    sum = 0
    for t_cls,p_cls in zip(actual_class,pred_class):
        if t_cls == p_cls:
           sum+= 1
    acc_test = sum / n_test
    print('Accuracy {0}'.format(acc_test))
writer.close()
