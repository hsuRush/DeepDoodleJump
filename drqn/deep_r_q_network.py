# -*- coding: UTF-8 -*-
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
import random
import numpy as np
from collections import deque
from tensorflow.contrib import rnn

#ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

IMAGE_W = 80
IMAGE_H = 80
IMAGE_C = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def RNN(args, x, n_hidden):
    #x = tf.unstack(x, 4, -1)
    weights = tf.Variable(tf.truncated_normal((n_hidden, args.actions), stddev = 0.01))
    biases = tf.constant(0.01, shape = (args.actions,))

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases
  

def createNetwork(args):
    # network weights
    W_conv1 = weight_variable([7, 7, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, args.actions])
    b_fc2 = bias_variable([args.actions])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 1])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)
    #print(tf.shape(h_conv3))
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 3200])

    #h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    #readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    # drqn readout layer
    readout = RNN(args, [h_conv3_flat], 128)
    #print(tf.shape(h_conv3_flat))
    #readout = tf.keras.layers.CuDNNLSTM(512 ,return_sequences=True) (h_conv3_flat)
    return s, readout, h_conv3_flat
