# -*- coding: UTF-8 -*-
from drqn.deep_r_q_network import createNetwork
from game.doodlejump import DoodleJump
from utils import preprocessing_image
import tensorflow as tf
from collections import deque
import os
import numpy as np
import cv2
import random
from argparse import ArgumentParser
import time 

parser = ArgumentParser(description='Train a ReID network.')
parser.add_argument(
    '--actions', default=3, type=int,
    help='Number of training iterations.')

parser.add_argument(
    '--exp_dir', default="exp_1", type=str,
    help='exp_dir')

parser.add_argument(
    '--iterations', default=100, type=int,
    help='test iterations')

parser.add_argument(
    '--fps', default=1000000, type=int,
    help='fps of the game')

args = parser.parse_args()

GAME = 'doodlejump' # the name of the game being played for log files
ACTIONS = args.actions # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

IMAGE_W = 80
IMAGE_H = 80
IMAGE_C = 1
def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = DoodleJump(args.fps)

    # store the previous observations in replay memory
    #D = deque()

    # printing
    if not os.path.exists( "logs_" + GAME ):
        os.mkdir("logs_" + GAME )
    
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    if IMAGE_C == 1:
        x_t = cv2.cvtColor(cv2.resize(x_t, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2GRAY)
    else:
        x_t = cv2.resize(x_t, (IMAGE_W, IMAGE_H))
    #ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.expand_dims(x_t,-1)

   
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(args.exp_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
     
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    top_score = 0
    total_score = 0
    start_time = time.time()
    f=0
    while t < args.iterations:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        
        if IMAGE_C == 1:
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2GRAY) 
            #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY) 
        else:
            x_t1 = cv2.resize(x_t1, (IMAGE_W, IMAGE_H))

    
        #s_t = s_t[:, :, 1:]

        #s_t1 = np.append(s_t, np.expand_dims(x_t1, axis=-1), axis=2) 
        #s_t = s_t1
        s_t1 = np.expand_dims(x_t1, -1)

        # update the old values
        s_t = s_t1 
        total_score += game_state.score
        if game_state.score > top_score:
            top_score = game_state.score

        if terminal:
            t += 1
        f+=1
    print("top score: ", top_score, end='')
    print(" in ", args.iterations, " iterations and", end='')
    print(" survived in ", str(time.time() - start_time), "seconds")
    print("avg score: ", total_score / f)

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(args)
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
