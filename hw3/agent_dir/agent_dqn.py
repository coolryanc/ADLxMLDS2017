from agent_dir.agent import Agent
from breakout_model import Model
import gym
from cv2 import resize, cvtColor, COLOR_RGB2GRAY, INTER_AREA
from collections import deque
import os
import random as ran
import datetime
import tensorflow as tf
import numpy as np

MINIBATCH_SIZE = 32
TRAIN_START = 10000
FINAL_EXPLORATION = 0.05
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
EXPLORATION = 1000000
START_EXPLORATION = 1.
DISCOUNT = 0.99
model_path = "./checkpoints_breakput/Breakout.ckpt"
env = gym.make('BreakoutNoFrameskip-v4')

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        if args.test_dqn:
            print('loading trained model')
            self.model = Model(tf.Session(), env.action_space.n)
            self.model.load_checkpoint()
            self.history = np.zeros([84, 84, 5], dtype=np.uint8)
        self.render = args.do_render

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=True):
        self.history[:, :, 4] = self.pre_proc(observation)
        self.history[:, :, :4] = self.history[:, :, 1:]
        # self.history[:, :, 4] = self.pre_proc(observation)
        # print(self.pre_proc(observation))
        # print(observation)
        # self.history[:, :, :4] = self.history[:, :, 1:]
        Q = self.model.get_q(self.history[:, :, :4])
        action = self.model.get_action(Q, 0.05)
        return action

    def train(self):
        with tf.Session() as sess:
            mainDQN = Model(sess, env.action_space.n, NAME='main')
            targetDQN = Model(sess, env.action_space.n, NAME='target')

            sess.run(tf.global_variables_initializer())

            # initial copy q_net -> target_net
            copy_ops = self.get_copy_var_ops(dest_scope_name="target",
                                        src_scope_name="main")
            sess.run(copy_ops)

            recent_rlist = deque(maxlen=100)
            e = 1.
            episode, epoch, frame = 0, 0, 0

            epoch_score, epoch_Q = deque(), deque()
            average_Q, average_reward = deque(), deque()

            epoch_on = False
            no_life_game = False
            replay_memory = deque(maxlen=MEMORY_SIZE)

            # Train agent during 200 epoch
            while epoch <= 200:
                episode += 1
                if self.render:
                    env.render()
                history = np.zeros([84, 84, 5], dtype=np.uint8)
                rall, count = 0, 0
                d = False
                ter = False
                start_lives = 0
                s = env.reset()

                while not d:
                    if self.render:
                        env.render()

                    frame += 1
                    count += 1

                    # e-greedy
                    if e > FINAL_EXPLORATION and frame > TRAIN_START:
                        e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                    Q = mainDQN.get_q(history[:, :, :4])
                    average_Q.append(np.max(Q))

                    action = mainDQN.get_action(Q, e)

                    s1, r, d, l = env.step(action)
                    ter = d
                    reward = np.clip(r, -1, 1)

                    no_life_game, start_lives = self.get_game_type(count, l, no_life_game, start_lives)

                    ter, start_lives = self.get_terminal(start_lives, l, reward, no_life_game, ter)

                    history[:, :, 4] = self.pre_proc(s1)

                    replay_memory.append((np.copy(history[:, :, :]), action, reward, ter))
                    history[:, :, :4] = history[:, :, 1:]

                    rall += r

                    if frame > TRAIN_START:

                        minibatch = ran.sample(replay_memory, MINIBATCH_SIZE)
                        self.train_minibatch(mainDQN, targetDQN, minibatch)

                        if frame % TARGET_UPDATE == 0:
                            copy_ops = self.get_copy_var_ops(dest_scope_name="target",
                                                        src_scope_name="main")
                            sess.run(copy_ops)

                    if (frame - TRAIN_START) % 50000 == 0:
                        epoch_on = True

                recent_rlist.append(rall)

                average_reward.append(rall)

                print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                      "Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}  ".format(episode, frame, count, rall, e,
                                                                            np.mean(average_Q),
                                                                            np.mean(recent_rlist)))
                fd = open('HistoryDQN.csv','a')
                fd.write('Episode: %d, Score: %f\n' % (episode, rall))
                fd.close()

                if epoch_on:
                    epoch += 1
                    self.saveModel(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN)
                    epoch_on = False
                    average_reward = deque()
                    average_Q = deque()

    def pre_proc(self, X):
        x = np.uint8(
            resize(cvtColor(X, COLOR_RGB2GRAY)*255, (84, 84)), interpolation=INTER_AREA)
        return x

    def get_copy_var_ops(self, *, dest_scope_name="target", src_scope_name="main"):
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def get_game_type(self, count, l, no_life_game, start_live):
        if count == 1:
            start_live = l['ale.lives']
            if start_live == 0:
                no_life_game = True
            else:
                no_life_game = False
        return [no_life_game, start_live]


    def get_terminal(self, start_live, l, reward, no_life_game, ter):
        if no_life_game:
            if reward < 0:
                ter = True
        else:
            if start_live > l['ale.lives']:
                ter = True
                start_live = l['ale.lives']
        return [ter, start_live]


    def train_minibatch(self, mainDQN, targetDQN, minibatch):
        s_stack = []
        a_stack = []
        r_stack = []
        s1_stack = []
        d_stack = []

        for s_r, a_r, r_r, d_r in minibatch:
            s_stack.append(s_r[:, :, :4])
            a_stack.append(a_r)
            r_stack.append(r_r)
            s1_stack.append(s_r[:, :, 1:])
            d_stack.append(d_r)

        d_stack = np.array(d_stack) + 0
        Q1 = targetDQN.get_q(np.array(s1_stack))
        y = r_stack + (1 - d_stack) * DISCOUNT * np.max(Q1, axis=1)
        mainDQN.sess.run(mainDQN.train, feed_dict={mainDQN.X: np.float32(np.array(s_stack) / 255.), mainDQN.Y: y,
                                                   mainDQN.a: a_stack})

    def saveModel(self, epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN):
        save_path = mainDQN.saver.save(mainDQN.sess, model_path, global_step=(epoch - 1))
        print("Model(epoch :", epoch, ") saved in file: ", save_path, " Now time : ", datetime.datetime.now())
