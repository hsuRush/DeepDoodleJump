# -*- coding: UTF-8 -*-
from dqn.deep_q_network import createNetwork
from game.doodlejump import DoodleJump
from utils import preprocessing_image
import tensorflow as tf
from collections import deque
import os
import numpy as np
import cv2
import random
from argparse import ArgumentParser

parser = ArgumentParser(description='Train a ReID network.')
parser.add_argument(
    '--actions', default=3, type=int,
    help='Number of training iterations.')

parser.add_argument(
    '--exp_dir', default="saved_network", type=str,
    help='exp_dir')

args = parser.parse_args()


GAME = 'doodlejump' # the name of the game being played for log files
ACTIONS = args.actions # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
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
    game_state = DoodleJump()

    # store the previous observations in replay memory
    D = deque()

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
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

   
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(args.exp_dir)
    t = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        t = int(checkpoint.model_checkpoint_path.spilt("-")[-1])

    else:
        print("Could not find old network weights")
        t = 0
    # start training
    epsilon = INITIAL_EPSILON
    #t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        if IMAGE_C == 1:
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2GRAY) 
            #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY) 
        else:
            x_t1 = cv2.resize(x_t1, (IMAGE_W, IMAGE_H))

    
        s_t = s_t[:, :, 1:]

        s_t1 = np.append(s_t, np.expand_dims(x_t1, axis=-1), axis=2) 
        s_t = s_t1
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 50000 == 0:
            saver.save(sess, args.exp_dir + '/' + GAME + '-drqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(args)
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
