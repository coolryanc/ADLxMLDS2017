import argparse
from agent_dir.agent import Agent
import scipy
import numpy as np
import gym
import pickle
from pong_model import Model



def prepro(observation):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    observation = observation[35:195]  # crop
    observation = observation[::2, ::2, 0]  # downsample by factor of 2
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    return observation.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        self.hidden_layer_size = 200
        self.learning_rate = 0.0005
        self.batch_size_episodes = 1
        self.checkpoint_every_n_episodes = 10
        self.env = env
        self.UP_ACTION = 2
        self.DOWN_ACTION = 3
        self.action_dict = {self.DOWN_ACTION: 0, self.UP_ACTION: 1}
        super(Agent_PG,self).__init__(env)
        if args.test_pg:
            self.model = Model(
                self.hidden_layer_size, self.learning_rate, checkpoints_dir='checkpoints')
            self.model.load_checkpoint()
        else:
            self.model = Model(
                self.hidden_layer_size, self.learning_rate, checkpoints_dir='checkpoints')



    def init_game_setting(self):
        # Mapping from action values to outputs from the policy network
        self.last_observation = self.env.reset()
        self.last_observation = prepro(self.last_observation)

    def train(self):
        batch_state_action_reward_tuples = []
        smoothed_reward = None
        episode_n = 1
        while True:
            print("Starting episode %d" % episode_n)
            episode_done = False
            episode_reward_sum = 0
            round_n = 1
            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            observation = last_observation
            n_steps = 1
            while not episode_done:
                observation_delta = observation - last_observation
                last_observation = observation
                up_probability = self.model.forward_pass(observation_delta)[0]
                if np.random.uniform() < up_probability:
                    action = self.UP_ACTION
                else:
                    action = self.DOWN_ACTION
                observation, reward, episode_done, info = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
                n_steps += 1

                tup = (observation_delta, self.action_dict[action], reward)
                batch_state_action_reward_tuples.append(tup)
                if reward != 0:
                    round_n += 1
                    n_steps = 0
            print("Episode %d finished after %d rounds" % (episode_n, round_n))

            # exponentially smoothed version of reward
            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Reward total was %.3f; discounted moving average of reward is %.3f" \
                % (episode_reward_sum, smoothed_reward))

            if episode_n % 10 == 0:
                self.model.save_checkpoint()
            episode_n += 1


    def make_action(self, observation, test=True):
        observation = prepro(observation)
        observation_delta = observation - self.last_observation
        self.last_observation = observation
        up_probability = self.model.forward_pass(observation_delta)[0]

        if np.random.uniform() < up_probability:
            action = self.UP_ACTION
        else:
            action = self.DOWN_ACTION

        return action
